import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, AutoConfig, LlamaTokenizer # LlamaTokenizer needed for actual use, AutoConfig for dummy Llama
from timm.models.vision_transformer import VisionTransformer
from torch.distributions.normal import Normal, Categorical
import numpy as np
from collections import deque, namedtuple
import random
import math
import copy

# --- ASI Configuration ---
class ASIConfig:
    def __init__(self):
        self.hid = 8192      # Hidden size
        self.exp = 64        # Number of experts (MoE)
        self.mem_c = 1000000 # Memory cells
        self.q_dim = 1024    # Quantum dimension
        self.e_dim = 512     # Emotion dimension
        self.eth_l = 16      # Ethics layers
        self.self_d = 32     # Self-model depth
        self.wm_h = 32       # World model heads
        self.wm_l = 16       # World model layers
        self.fus_h = 16      # Fusion heads
        self.fus_l = 4       # Fusion layers

        self.lr_s = 3e-5     # Learning rate start
        self.lr_e = 1e-6     # Learning rate end
        self.g_norm = 1.0    # Max grad norm
        self.plast_s = 0.05  # Plasticity start
        self.plast_e = 0.005 # Plasticity end
        self.gamma = 0.99    # RL discount factor
        self.lambda_gae = 0.95 # GAE lambda
        self.ent_c = 0.01    # Entropy coefficient
        self.val_c = 0.5     # Value loss coefficient
        self.tgt_upd = 1000  # Target network update interval

        self.per_a = 0.6     # PER alpha
        self.per_b_s = 0.4   # PER beta start
        self.per_b_f = 1000000 # PER beta frames

        self.meta_lr = 1e-3  # Meta-learning inner LR
        self.meta_is = 5     # Meta-learning inner steps

        self.cont_t = 0.07   # Contrastive temperature
        self.wm_c = 0.5      # World model loss coefficient
        self.meta_c = 0.1    # Meta loss coefficient
        self.cont_c = 0.1    # Contrastive loss coefficient
        self.cur_c = 0.05    # Curiosity loss coefficient
        self.ewc_l = 0.1     # EWC lambda
        self.l2_reg = 1e-5   # L2 regularization coefficient

        self.adv_c = 0.01 # Adversarial loss coefficient (for self-challenging)
        self.res_c = 0.001 # Resource allocation loss coefficient

        # Resource management parameters
        self.compute_budget = 1.0 # Normalized total compute available
        self.memory_budget = 1.0  # Normalized total memory available

# --- ASI Modules ---
class DynamicLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.activation = nn.GELU()

    def forward(self, x, skip_connection_weight=None):
        if skip_connection_weight is None:
            return self.activation(F.linear(x, self.weight, self.bias))
        else:
            return self.activation(F.linear(x, self.weight * skip_connection_weight, self.bias))

class QuantumMindLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj_r = nn.Linear(cfg.hid, cfg.q_dim)
        self.proj_i = nn.Linear(cfg.hid, cfg.q_dim)
        self.fuse_g = nn.Linear(cfg.q_dim * 2, cfg.hid)
        self.act = nn.GELU()

    def forward(self, x):
        real = self.act(self.proj_r(x))
        imag = self.act(self.proj_i(x))
        return self.fuse_g(torch.cat((real, imag), dim=-1))

class NeuroMemory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.episodic_k = nn.Parameter(torch.randn(cfg.mem_c // 2, cfg.hid))
        self.episodic_v = nn.Parameter(torch.randn(cfg.mem_c // 2, cfg.hid))
        self.semantic_k = nn.Parameter(torch.randn(cfg.mem_c // 2, cfg.hid))
        self.semantic_v = nn.Parameter(torch.randn(cfg.mem_c // 2, cfg.hid))
        self.working_m = nn.Parameter(torch.randn(1, 10, cfg.hid))
        self.att_e = nn.MultiheadAttention(cfg.hid, 16, batch_first=True)
        self.att_s = nn.MultiheadAttention(cfg.hid, 16, batch_first=True)
        self.att_w = nn.MultiheadAttention(cfg.hid, 8, batch_first=True)
        self.read_proj = nn.Linear(cfg.hid * 3, cfg.hid)
        self.reg_buf('episodic_u', torch.zeros(cfg.mem_c // 2))
        self.reg_buf('semantic_u', torch.zeros(cfg.mem_c // 2))

    def write(self, data, m_type='episodic', importance=1.0):
        if m_type == 'episodic': mk, mv, ut = self.episodic_k, self.episodic_v, self.episodic_u
        elif m_type == 'semantic': mk, mv, ut = self.semantic_k, self.semantic_v, self.semantic_u
        else: return # Working memory is updated via attention

        scores = torch.matmul(data, mk.T)
        probs = F.softmax(scores, dim=-1)
        upd_str = (probs * self.cfg.plast_s * importance).unsqueeze(-1) # Use actual plasticity

        mk.data = mk.data * (1 - upd_str.mean(dim=0)) + data.unsqueeze(1) * upd_str.mean(dim=0)
        mv.data = mv.data * (1 - upd_str.mean(dim=0)) + data.unsqueeze(1) * upd_str.mean(dim=0)
        ut.data = ut.data * 0.99 + probs.sum(dim=0).detach()

    def read(self, q):
        e_r, _ = self.att_e(q, self.episodic_k.unsqueeze(0).repeat(q.shape[0],1,1), self.episodic_v.unsqueeze(0).repeat(q.shape[0],1,1))
        s_r, _ = self.att_s(q, self.semantic_k.unsqueeze(0).repeat(q.shape[0],1,1), self.semantic_v.unsqueeze(0).repeat(q.shape[0],1,1))
        w_r, _ = self.att_w(q, self.working_m.repeat(q.shape[0],1,1), self.working_m.repeat(q.shape[0],1,1))
        return self.read_proj(torch.cat((e_r, s_r, w_r), dim=-1))

class ConsciousProcessor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_net = nn.TransformerEncoder(nn.TransformerEncoderLayer(cfg.hid, 16, 4*cfg.hid), cfg.self_d)
        self.self_proj = nn.Linear(cfg.hid, cfg.hid)
        self.img_gru = nn.GRU(cfg.hid, cfg.hid, 3)
        self.real_chk = nn.Linear(cfg.hid, cfg.hid)
        self.meta_eval = nn.Linear(cfg.hid, 4) # Confidence, Importance, Novelty, Uncertainty
        self.meta_proj = nn.Linear(4, cfg.hid)
        self.resource_allocator = nn.Linear(cfg.hid + 4, 2) # Output for compute, memory allocation

    def forward(self, x):
        self_r = self.self_net(x)
        img_s, _ = self.img_gru(x)
        real_s = self.real_chk(x)
        proc_x = x + 0.3*self.self_proj(self_r) + 0.3*img_s + 0.4*real_s
        meta_raw = self.meta_eval(proc_x.mean(dim=1))
        meta = F.sigmoid(meta_raw)
        proc_x = proc_x * (1 + self.meta_proj(meta.unsqueeze(1)))
        
        res_input = torch.cat((proc_x.mean(dim=1), meta), dim=-1)
        resource_alloc = F.softmax(self.resource_allocator(res_input), dim=-1) # Compute, Memory

        return proc_x, meta, resource_alloc

class EmotionEthicsSystem(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emo_enc = nn.Sequential(nn.Linear(cfg.hid, cfg.e_dim), nn.LeakyReLU(), nn.Linear(cfg.e_dim, 12))
        self.eth_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(cfg.hid, 8, 2*cfg.hid), cfg.eth_l)
        self.eth_gate = nn.Linear(cfg.hid + 12, cfg.hid)
        self.int_motiv = nn.Linear(cfg.hid + 12, 1)

    def forward(self, x):
        emotions_raw = self.emo_enc(x.mean(dim=1))
        emotions = F.softmax(emotions_raw, dim=-1)
        eth_emb = self.eth_enc(x)
        gated_in = torch.cat((eth_emb, emotions.unsqueeze(1).expand(-1, eth_emb.shape[1], -1)), dim=-1)
        output = F.sigmoid(self.eth_gate(gated_in)) * x
        int_rew_in = torch.cat((x.mean(dim=1), emotions), dim=-1)
        int_reward = self.int_motiv(int_rew_in).squeeze(-1)
        return output, emotions, int_reward

class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.trans = nn.TransformerEncoder(nn.TransformerEncoderLayer(cfg.hid, cfg.wm_h, 4*cfg.hid), cfg.wm_l)
        self.s_p_mu = nn.Linear(cfg.hid, cfg.hid)
        self.s_p_lv = nn.Linear(cfg.hid, cfg.hid)
        self.r_p = nn.Linear(cfg.hid, 1)
        self.oa_p = nn.Linear(cfg.hid, cfg.hid) # Other agent action
        self.causal_inf = nn.Sequential(nn.Linear(cfg.hid * 2, cfg.hid), nn.ReLU(), nn.Linear(cfg.hid, 1))
        self.adv_critic = nn.Linear(cfg.hid * 2, 1) # For adversarial self-challenging

    def forward(self, cur_s, act_t=None, oa_s=None):
        x = cur_s
        if act_t is not None: x = x + act_t if x.shape[1] == act_t.shape[1] else x + act_t.unsqueeze(1)
        if oa_s is not None: x = x + oa_s if x.shape[1] == oa_s.shape[1] else x + oa_s.unsqueeze(1)

        enc = self.trans(x)
        enc_m = enc.mean(dim=1)

        mu_ns = self.s_p_mu(enc_m)
        lv_ns = self.s_p_lv(enc_m)
        pred_r = self.r_p(enc_m).squeeze(-1)
        pred_oa = self.oa_p(enc_m)
        causal_in = torch.cat((enc_m, mu_ns), dim=-1)
        causal_s = self.causal_inf(causal_in).squeeze(-1)
        
        adv_in = torch.cat((enc_m, mu_ns), dim=-1)
        adv_score = self.adv_critic(adv_in).squeeze(-1)

        return mu_ns, lv_ns, pred_r, pred_oa, causal_s, adv_score

class MultimodalFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vis_p = nn.Linear(cfg.hid, cfg.hid)
        self.lang_p = nn.Linear(cfg.hid, cfg.hid)
        self.fuse_t = nn.TransformerEncoder(nn.TransformerEncoderLayer(cfg.hid, cfg.fus_h), cfg.fus_l)
        self.cls_t = nn.Parameter(torch.randn(1, 1, cfg.hid))

    def forward(self, vis_feats, lang_feats):
        if vis_feats.dim() == 2: vis_feats = vis_feats.unsqueeze(1)
        if lang_feats.dim() == 2: lang_feats = lang_feats.unsqueeze(1)

        v_p = self.vis_p(vis_feats)
        l_p = self.lang_p(lang_feats)
        cls_t = self.cls_t.expand(v_p.shape[0], -1, -1)
        fused_in = torch.cat((cls_t, v_p, l_p), dim=1)
        fused_out = self.fuse_t(fused_in)
        return fused_out[:, 0, :]

# --- Main ASI Architecture ---
class AdvancedASI(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = ASIConfig()
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sensors
        self.vision = VisionTransformer(img_size=384, patch_size=32, embed_dim=self.cfg.hid, num_classes=0)
        llama_cfg = AutoConfig.from_pretrained("HuggingFaceM4/tiny-random-LlamaForCausalLM", hidden_size=self.cfg.hid, num_hidden_layers=2, num_attention_heads=4)
        self.language = LlamaForCausalLM(llama_cfg)
        self.language.eval()

        # Core Cognitive Modules
        self.mult_f = MultimodalFusion(self.cfg)
        self.q_layer = QuantumMindLayer(self.cfg)
        self.mem = NeuroMemory(self.cfg)
        self.con = ConsciousProcessor(self.cfg)
        self.emo_eth = EmotionEthicsSystem(self.cfg)
        self.wm = WorldModel(self.cfg)

        # Executive Systems
        self.dec_pol = nn.TransformerDecoder(nn.TransformerDecoderLayer(self.cfg.hid, 16, 4*self.cfg.hid), 8)
        self.act_h = nn.Linear(self.cfg.hid, self.cfg.hid) # Action space size
        self.crit_h = nn.Linear(self.cfg.hid, 1)

        # Self-Improvement & Meta-learning
        self.opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr_s)
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        self.plast = nn.Parameter(torch.ones(1) * self.cfg.plast_s)
        self.tgt_net = copy.deepcopy(self).to(self.dev)
        self.tgt_net.eval()
        self.ewc_p = {} # Fisher Info & old params

        # Recursive self-modification (conceptual)
        self.self_mod_net = nn.Sequential(
            nn.Linear(self.cfg.hid * 2, self.cfg.hid), # Input: current state + meta_info
            nn.GELU(),
            nn.Linear(self.cfg.hid, self.cfg.hid // 16), # Output: small vector for modification
            nn.GELU(),
            nn.Linear(self.cfg.hid // 16, 2) # Example: modify plasticity/learning rate
        )

        # Dynamic Module Selection (MoE inspired - conceptual)
        self.module_router = nn.Linear(self.cfg.hid, len(self.modules())-1) # Router based on context

        self.to(self.dev)

    def forward(self, images, texts, params=None):
        if images.dim() == 2: v_f = images.unsqueeze(1)
        else: v_f = self.vision(images).unsqueeze(1)

        if isinstance(texts, dict): l_out = self.language(**texts).last_hidden_state.mean(dim=1)
        else: l_out = texts
            
        f_in = self.mult_f(v_f, l_out).unsqueeze(1)

        # Dynamic Module Selection
        # This is a highly conceptual implementation for MoE-like routing
        # In a real scenario, this would route input to specific sub-modules
        # Here, it's a simple dynamic weighting of core modules' outputs
        
        # Calculate router scores for main processing branches (conceptual)
        # For simplicity, we'll just weight the primary outputs.
        # A more complex system would route the input 'f_in' to different sub-modules
        # based on these scores and aggregate their outputs.
        router_scores = F.softmax(self.module_router(f_in.mean(dim=1)), dim=-1)
        
        # Q-Layer
        q_out = self.q_layer(f_in)
        
        # Memory
        imp = 1.0 # Placeholder
        self.mem.write(q_out.detach().squeeze(1), 'episodic', imp)
        self.mem.write(q_out.detach().squeeze(1), 'semantic', imp)
        m_r = self.mem.read(q_out)
        mem_q_out = q_out + m_r

        # Consciousness
        con_out, meta_inf, res_alloc = self.con(mem_q_out)
        
        # Emotion & Ethics
        ee_out, emotions, int_rew = self.emo_eth(con_out)

        # World Model
        mu_ns, lv_ns, pred_r, pred_oa, causal_s, adv_s = self.wm(ee_out)

        # Decision
        dec_ctx = self.dec_pol(ee_out, ee_out).squeeze(1) # Using ee_out as both src/tgt for decoder
        act_log = self.act_h(dec_ctx)
        s_val = self.crit_h(dec_ctx).squeeze(-1)

        # Conceptual dynamic weighting based on router scores
        # Example: if router_scores prioritize 'emotional' processing for this input
        # final_decision_influence = router_scores[:, 0] * act_log + router_scores[:, 1] * (act_log * emotions_influence_matrix)
        # This requires more specific design for each module's contribution

        return act_log, s_val, pred_r, mu_ns, lv_ns, emotions, meta_inf, int_rew, causal_s, adv_s, res_alloc

    def self_improve(self, loss_val, metrics_dict, step):
        self.plast.data = torch.clamp(self.plast * (1 - (loss_val / (loss_val + 1.0)) * 0.1) * (1 + 0.05 * torch.randn(1).to(self.dev)), self.cfg.plast_e, self.cfg.plast_s)
        
        # Recursive self-modification of internal parameters
        # Input to self_mod_net could be current state + meta_info
        dummy_state = torch.randn(1, self.cfg.hid).to(self.dev) # A simplified input
        mod_params = self.self_mod_net(torch.cat((dummy_state, metrics_dict['meta_info'].mean(dim=0)), dim=-1))
        # Example: use mod_params to adjust self.plast.data or self.cfg.lr_s dynamically
        self.plast.data *= F.sigmoid(mod_params[:,0]).item() # Example: modulate plasticity
        
        if step % self.cfg.tgt_upd == 0: self.update_target_network()

    def update_target_network(self): self.tgt_net.load_state_dict(self.state_dict())

    def recursive_self_enhancement(self, steps=1):
        for _ in range(steps):
            fake_i = torch.randn(1, 3, 384, 384).to(self.dev)
            fake_t = {'input_ids': torch.randint(0, self.language.config.vocab_size, (1, 10)).to(self.dev),
                      'attention_mask': torch.ones((1, 10), dtype=torch.long).to(self.dev)}
            self.train()
            _, _, pred_r, _, _, _, _, int_rew, _, _, _ = self(fake_i, fake_t)
            int_loss = -pred_r.mean() - int_rew.mean() + self.regularization_loss() + self.ewc_loss()
            self.opt.zero_grad()
            if self.scaler: self.scaler.scale(int_loss).backward(); self.scaler.step(self.opt); self.scaler.update()
            else: int_loss.backward(); self.opt.step()
        self.eval()

    def metacognition_report(self):
        params = sum(p.numel() for p in self.parameters())
        print(f"--- Meta Report ---")
        print(f"Params: {params/1e9:.3f}B | Plast: {self.plast.item():.5f} | LR: {self.opt.param_groups[0]['lr']:.8f}")

    def regularization_loss(self):
        l2 = 0.0
        for n, p in self.named_parameters():
            if 'weight' in n: l2 += torch.norm(p, p=2)
        return self.cfg.l2_reg * l2

    def ewc_loss(self):
        if not self.ewc_p: return 0.0
        loss = 0.0
        for n, p in self.named_parameters():
            if n in self.ewc_p: loss += (self.ewc_p[n]['fisher'] * (p - self.ewc_p[n]['old_param']).pow(2)).sum()
        return self.cfg.ewc_l * loss

    def consolidate_ewc_params(self):
        self.eval()
        f_info = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
        o_p = {n: p.clone().detach() for n, p in self.named_parameters() if p.requires_grad}
        d_i = torch.randn(1, 3, 384, 384).to(self.dev)
        d_t = {'input_ids': torch.randint(0, self.language.config.vocab_size, (1, 10)).to(self.dev),
               'attention_mask': torch.ones((1, 10), dtype=torch.long).to(self.dev)}
        act_l, _, _, _, _, _, _, _, _, _, _ = self(d_i, d_t)
        d_l = F.log_softmax(act_l, dim=-1).mean()
        self.zero_grad()
        d_l.backward()
        for n, p in self.named_parameters():
            if p.grad is not None: f_info[n] += p.grad.data.pow(2)
        self.ewc_p = {n: {'fisher': f_info[n], 'old_param': o_p[n]} for n in f_info}

# --- Autonomous Learner ---
class AutonomousLearner:
    def __init__(self, asi: AdvancedASI):
        self.asi = asi
        self.cfg = asi.cfg
        self.Transition = namedtuple('Transition', ('s_img', 's_text', 'act', 'rew_ext', 'ns_img', 'ns_text', 'td_err', 'is_fin', 'rew_int'))
        self.buf = deque(maxlen=self.cfg.mem_c)
        self.steps = 0

    def interact(self, env, num_steps=100):
        s_img, s_text = env.reset()
        for _ in range(num_steps):
            self.asi.eval()
            with torch.no_grad():
                act_l, s_val, pred_r, _, _, _, _, int_rew_p, _, _, _ = self.asi(s_img.to(self.asi.dev), s_text.to(self.asi.dev))
                act_d = Categorical(F.softmax(act_l.squeeze(0), dim=-1))
                act = act_d.sample()

            ns_img, ns_text, rew, done, _ = env.step(act.item())
            
            self.asi.tgt_net.eval()
            with torch.no_grad():
                _, ns_val_tgt, _, _, _, _, _, _, _, _, _ = self.asi.tgt_net(ns_img.to(self.asi.dev), ns_text.to(self.asi.dev))
                comb_rew = rew + int_rew_p.item() * self.cfg.cur_c
                ns_val_tgt[done] = 0.0
                td_e = (comb_rew + self.cfg.gamma * ns_val_tgt - s_val).abs().item()

            self.buf.append(self.Transition(s_img.cpu(), s_text.cpu(), act.cpu(), rew, ns_img.cpu(), ns_text.cpu(), td_e, done, int_rew_p.cpu()))
            s_img, s_text = ns_img, ns_text
            if done: s_img, s_text = env.reset()
            self.learn()

    def learn(self, bs=128):
        if len(self.buf) < bs * 2: return
        self.asi.train()
        self.asi.opt.zero_grad()

        batch, idx, is_w = self.per_sample(bs)
        s_i, s_t, acts, r_ext, ns_i, ns_t, _, is_f, r_int_buf = zip(*batch)

        s_i = torch.cat(s_i).to(self.asi.dev)
        s_t_ids = torch.cat([t['input_ids'] for t in s_t]).to(self.asi.dev)
        s_t_att = torch.cat([t['attention_mask'] for t in s_t]).to(self.asi.dev)
        s_t_proc = {'input_ids': s_t_ids, 'attention_mask': s_t_att}

        acts = torch.cat(acts).to(self.asi.dev)
        r_ext = torch.tensor(r_ext, dtype=torch.float32).to(self.asi.dev)
        r_int_buf = torch.cat(r_int_buf).to(self.asi.dev)

        ns_i = torch.cat(ns_i).to(self.asi.dev)
        ns_t_ids = torch.cat([t['input_ids'] for t in ns_t]).to(self.asi.dev)
        ns_t_att = torch.cat([t['attention_mask'] for t in ns_t]).to(self.asi.dev)
        ns_t_proc = {'input_ids': ns_t_ids, 'attention_mask': ns_t_att}
        is_f = torch.tensor(is_f, dtype=torch.bool).to(self.asi.dev)
        is_w = torch.tensor(is_w, dtype=torch.float32).to(self.asi.dev)

        with torch.cuda.amp.autocast(enabled=self.asi.scaler is not None):
            act_l, s_val, p_r_wm, mu_ns, lv_ns, emotions, meta_inf, int_r_wm, causal_s, adv_s, res_alloc = \
                self.asi(s_i, s_t_proc)

            with torch.no_grad():
                ns_fused_feat = self.asi.mult_f(self.asi.vision(ns_i).unsqueeze(1), self.asi.language(**ns_t_proc).last_hidden_state.mean(dim=1))
                _, ns_val_tgt, _, _, _, _, _, _, _, _, _ = self.asi.tgt_net(ns_i, ns_t_proc)
                ns_val_tgt[is_f] = 0.0

            r_comb = r_ext + int_r_wm * self.cfg.cur_c
            td_t = r_comb + self.cfg.gamma * ns_val_tgt
            adv = (td_t - s_val).detach()

            # Losses
            v_l = F.mse_loss(s_val, td_t.detach()) * self.cfg.val_c
            act_d = Categorical(F.softmax(act_l, dim=-1))
            log_p = act_d.log_prob(acts.squeeze())
            pol_l = -(log_p * adv * is_w).mean()
            ent_l = -act_d.entropy().mean() * self.cfg.ent_c

            wm_rec_l = -Normal(mu_ns, torch.exp(0.5 * lv_ns)).log_prob(ns_fused_feat).mean()
            wm_r_l = F.mse_loss(p_r_wm, r_ext)
            wm_causal_l = F.binary_cross_entropy_with_logits(causal_s, torch.ones_like(causal_s))
            wm_l = wm_rec_l + wm_r_l + wm_causal_l

            meta_l = self.meta_learn(s_i, s_t_proc)
            cur_l = F.mse_loss(int_r_wm, r_int_buf) # ASI learns to predict its own intrinsic reward
            cont_l = self.cont_learn(self.asi.mult_f(self.asi.vision(s_i).unsqueeze(1), self.asi.language(**s_t_proc).last_hidden_state.mean(dim=1)), ns_fused_feat)
            ewc_l_val = self.asi.ewc_loss()

            # Adversarial self-challenging (conceptual: ASI attempts to predict a "challenge" and minimize it)
            adv_l = F.binary_cross_entropy_with_logits(adv_s, torch.zeros_like(adv_s)) # Minimize adversarial score

            # Dynamic Resource Allocation Loss: penalize deviation from budget
            # res_alloc: [compute_ratio, memory_ratio]
            resource_loss = F.mse_loss(res_alloc[:, 0], torch.tensor(self.cfg.compute_budget, device=self.asi.dev)) + \
                            F.mse_loss(res_alloc[:, 1], torch.tensor(self.cfg.memory_budget, device=self.asi.dev))


            total_l = (pol_l + v_l + ent_l + self.cfg.wm_c * wm_l + self.cfg.meta_c * meta_l +
                       self.cfg.cont_c * cont_l + self.cfg.cur_c * cur_l + self.asi.regularization_loss() +
                       ewc_l_val + self.cfg.adv_c * adv_l + self.cfg.res_c * resource_loss)

        if self.asi.scaler:
            self.asi.scaler.scale(total_l).backward()
            self.asi.scaler.unscale_(self.asi.opt)
            torch.nn.utils.clip_grad_norm_(self.asi.parameters(), max_norm=self.cfg.g_norm * (1 + math.log(1 + self.steps/1000)))
            self.asi.scaler.step(self.asi.opt)
            self.asi.scaler.update()
        else:
            total_l.backward()
            torch.nn.utils.clip_grad_norm_(self.asi.parameters(), max_norm=self.cfg.g_norm * (1 + math.log(1 + self.steps/1000)))
            self.asi.opt.step()

        self.adapt_hp(total_l, self.steps)
        self.update_mem_priorities(batch, idx, s_i, s_t_proc, acts, r_comb, ns_i, ns_t_proc, is_f)

        if self.steps % 1000 == 0:
            self.diagnose(meta_inf)
            self.asi.metacognition_report()
            self.asi.recursive_self_enhancement(steps=1)

        self.steps += 1

    def per_sample(self, bs):
        beta = min(1.0, self.cfg.per_b_s + self.steps * (1.0 - self.cfg.per_b_s) / self.cfg.per_b_f)
        p = np.array([t.td_err for t in self.buf])
        if len(p) == 0: return [], [], []
        pr = (p + 1e-6) ** self.cfg.per_a
        pr /= pr.sum()
        idx = np.random.choice(len(self.buf), bs, p=pr, replace=False)
        batch = [self.buf[i] for i in idx]
        N = len(self.buf)
        is_w = (N * pr[idx]) ** (-beta)
        is_w /= is_w.max()
        return batch, idx, is_w

    def meta_learn(self, s_i, s_t_proc):
        t_asi = copy.deepcopy(self.asi).to(self.asi.dev)
        t_opt = torch.optim.AdamW(t_asi.parameters(), lr=self.cfg.meta_lr)
        for _ in range(self.cfg.meta_is):
            t_asi.train()
            t_opt.zero_grad()
            _, t_s_val, t_p_r, t_mu_ns, _, _, _, _, _, _, _ = t_asi(s_i, s_t_proc)
            m_i_l = F.mse_loss(t_mu_ns, t_s_val.unsqueeze(-1).expand_as(t_mu_ns))
            m_i_l.backward(); t_opt.step()
        self.asi.eval()
        with torch.no_grad():
            _, t_s_val_m, t_p_r_m, t_mu_ns_m, _, _, _, _, _, _, _ = t_asi(s_i, s_t_proc)
            m_l = -t_p_r_m.mean() + F.mse_loss(t_mu_ns_m, t_s_val_m.unsqueeze(-1).expand_as(t_mu_ns_m))
        return m_l

    def cont_learn(self, s_fused, ns_fused):
        s_n = F.normalize(s_fused, dim=1)
        ns_n = F.normalize(ns_fused, dim=1)
        logits = torch.matmul(s_n, ns_n.T) / self.cfg.cont_t
        labels = torch.arange(len(s_n), device=logits.dev)
        return F.cross_entropy(logits, labels)

    def adapt_hp(self, total_l, step):
        lr_decay = 0.5 * (1 + math.cos(math.pi * step / self.cfg.per_b_f))
        new_lr = self.cfg.lr_s + (self.cfg.lr_e - self.cfg.lr_s) * lr_decay
        for pg in self.asi.opt.param_groups: pg['lr'] = new_lr

    def update_mem_priorities(self, batch, idx, s_i, s_t, acts, r_comb, ns_i, ns_t, is_f):
        self.asi.eval()
        with torch.no_grad():
            _, ns_v_t, _, _, _, _, _, _, _, _, _ = self.asi.tgt_net(ns_i, ns_t)
            _, curr_s_v, _, _, _, _, _, _, _, _, _ = self.asi(s_i, s_t)
            ns_v_t[is_f] = 0.0
            td_e_new = (r_comb + self.cfg.gamma * ns_v_t - curr_s_v).abs().cpu().numpy()
        for i, j in enumerate(idx):
            t = self.buf[j]
            self.buf[j] = self.Transition(t.s_img, t.s_text, t.act, t.rew_ext, t.ns_img, t.ns_text, td_e_new[i], t.is_fin, t.rew_int)

    def diagnose(self, meta_info):
        cons = meta_info[:, 0].std().item() / (meta_info[:, 0].mean().abs().item() + 1e-9) # Meta-info for consistency
        conf = meta_info[:, 0].mean().item() # Confidence
        nov = meta_info[:, 2].mean().item() # Novelty
        if cons > 0.8 or conf < 0.3: print("D: High inconsistency/low confidence. More exploration.")
        if nov < 0.2: print("D: Low novelty. Increase exploration.")
        # Adjust entropy if needed
        # self.asi.cfg.ent_c = max(self.asi.cfg.ent_c * 0.95, 0.005) if conf > 0.7 else min(self.asi.cfg.ent_c * 1.05, 0.05)


# --- Dummy Environment ---
class DummyEnvironment:
    def __init__(self, state_dim=8192, action_dim=10):
        self.sd = state_dim
        self.ad = action_dim
        self.cs_img = None
        self.cs_text = None
        self.steps = 0

    def reset(self):
        self.cs_img = torch.randn(1, 3, 384, 384)
        self.cs_text = {'input_ids': torch.randint(0, 30000, (1, 50)), 'attention_mask': torch.ones((1, 50), dtype=torch.long)}
        self.steps = 0
        return self.cs_img, self.cs_text

    def step(self, action):
        n_img = self.cs_img + 0.1 * torch.randn_like(self.cs_img)
        n_text_ids = (self.cs_text['input_ids'] + torch.randint(-5, 5, self.cs_text['input_ids'].shape)).clamp(0, 30000)
        n_text = {'input_ids': n_text_ids, 'attention_mask': self.cs_text['attention_mask']}
        self.cs_img, self.cs_text = n_img, n_text
        rew = random.random() * 2 - 1
        self.steps += 1
        done = self.steps >= 100 or random.random() < 0.01
        return self.cs_img, self.cs_text, rew, done, {}

# --- Example Usage ---
if __name__ == "__main__":
    asi_model = AdvancedASI()
    print("ASI Model Initialized!")
    asi_model.metacognition_report()

    env = DummyEnvironment(state_dim=asi_model.cfg.hid, action_dim=asi_model.act_h.out_features)
    learner = AutonomousLearner(asi_model)

    print("\nStarting Autonomous Interaction & Learning...")
    for ep in range(2):
        print(f"\n--- Episode {ep + 1} ---")
        learner.interact(env, num_steps=100)
        
        if (ep + 1) % 1 == 0: asi_model.consolidate_ewc_params()

        if (ep + 1) % 1 == 0: asi_model.metacognition_report()

    print("\nAutonomous Learning Complete.")
    print("Final Metacognition Report:")
    asi_model.metacognition_report()

    dummy_i = torch.randn(1, 3, 384, 384).to(asi_model.dev)
    dummy_t = {'input_ids': torch.randint(0, asi_model.language.config.vocab_size, (1, 20)).to(asi_model.dev),
               'attention_mask': torch.ones((1, 20), dtype=torch.long).to(asi_model.dev)}

    asi_model.eval()
    with torch.no_grad():
        act_l, s_val, p_r, mu_ns, lv_ns, emo, meta, int_r, causal_s, adv_s, res_alloc = asi_model(dummy_i, dummy_t)
        print("\n--- Single Forward Pass Example ---")
        print(f"Action Logits Shape: {act_l.shape}")
        print(f"State Value Shape: {s_val.shape}")
        print(f"Predicted Ext Reward: {p_r.item():.4f}")
        print(f"Predicted Int Reward: {int_r.item():.4f}")
        print(f"Meta Info (Conf, Imp, Nov, Unc): {meta.tolist()}")
        print(f"Resource Allocation (Compute, Memory): {res_alloc.tolist()}")