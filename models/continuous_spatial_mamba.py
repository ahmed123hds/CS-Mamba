import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContinuousSpatialSSM(nn.Module):
    """
    Physics-Informed Continuous Spatial PDE Mamba (PINN-CS-Mamba)
    =============================================================
    Instead of an arbitrary learned Filter (which Spatial-Mamba does), 
    this explicitly enforces true Thermodynamic Diffusion (The Heat Equation).
    It physically links adjacent elements using a Hard-Coded Mathematical Laplacian
    Operator (∇²) multiplied by a single Learnable Diffusivity Constant per channel.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        d_inner = int(expand * d_model)
        
        # Dual Input-dependent Time-Scales (Gates)
        self.dt_self_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.dt_diff_proj = nn.Linear(d_inner, d_inner, bias=True)
        
        # --- CRITICAL ODE STABILITY INITIALIZATION ---
        dt_init = math.log(math.exp(0.1) - 1.0)
        nn.init.constant_(self.dt_self_proj.bias, dt_init)
        nn.init.constant_(self.dt_diff_proj.bias, dt_init)
        nn.init.uniform_(self.dt_self_proj.weight, -1e-4, 1e-4)
        nn.init.uniform_(self.dt_diff_proj.weight, -1e-4, 1e-4)

        # Mamba Projections
        self.B_proj  = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj  = nn.Linear(d_inner, d_state, bias=False)
        self.D       = nn.Parameter(torch.ones(d_inner))
        
        # S4 / Mamba Log-decay initialization
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).unsqueeze(0).expand(d_inner, -1))

        # --- PHYSICS-INFORMED THERMODYNAMIC DIFFUSION ---
        # Fixed Mathematical Laplacian ∇² Operator (Non-Learnable physics constraint)
        laplacian_kernel = torch.tensor([
            [0.0,  1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0,  1.0, 0.0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Expand across D channels for independent depthwise application
        laplacian_kernel = laplacian_kernel.repeat(d_inner, 1, 1, 1)
        self.register_buffer("laplacian", laplacian_kernel)
        
        # Learnable Diffusivity Constant (D) - Initialized to 0 for stability
        self.diffusivity_raw = nn.Parameter(torch.zeros(1, d_inner, 1, 1))

    def forward(self, x: torch.Tensor, K_steps: int = 3) -> torch.Tensor:
        B_val, N, D_dim = x.shape
        H = W = int(math.sqrt(N))
        assert H * W == N, "Spatial Mamba requires N to be a perfect square."

        dt = 1.0 / K_steps

        # Enforce Re(lambda) < 0 for stability
        A_mat = -F.softplus(self.A_log)           # (D, S) < 0
        
        # Rigorous Euler Bound Clamp
        delta_self = torch.clamp(F.softplus(self.dt_self_proj(x)), max=0.15)  # (B, N, D)
        delta_diff = torch.clamp(F.softplus(self.dt_diff_proj(x)), max=0.15)  # (B, N, D)
        
        B_mat = self.B_proj(x)                    # (B, N, S)
        C_mat = self.C_proj(x)                    # (B, N, S)

        # 3. Initialize PDE State: h(0) = B(x) * x
        h = torch.einsum('bnd,bns->bnds', x, B_mat)

        # 4. Explicit Euler Spatial Diffusion Loop
        for _ in range(K_steps):
            
            # --- THERMODYNAMIC DIFFUSION ENGINE ---
            # Collapse S-state temporarily for unified physical routing
            h_spatial = h.sum(dim=-1) 
            h_2d = h_spatial.transpose(1, 2).view(B_val, D_dim, H, W) 
            
            # Step 1: Calculate Spatial Gradient using True Laplacian (∇²h)
            laplacian_h = F.conv2d(h_2d, self.laplacian, padding=1, groups=D_dim)
            
            # Step 2: Apply Physical Diffusivity Parameter (D * ∇²h)
            # Physics constraint: D MUST be > 0. If D < 0, it causes inverse-diffusion
            # which guarantees an instant gradient explosion. Max bound 0.5 for CFL.
            D_phys = torch.sigmoid(self.diffusivity_raw) * 0.5
            diffused_h_2d = D_phys * laplacian_h
            
            # Reflatten to sequence (B, N, D) and expand state (B, N, D, S)
            diffused_h = diffused_h_2d.view(B_val, D_dim, N).transpose(1, 2)
            diffused_h = diffused_h.unsqueeze(-1) # Broadcast back across State dimension

            # --- DUAL GATED PDE STEP ---
            # Force 1: Internal Mamba Self-Decay (A) + Input (B)
            mamba_decay = torch.einsum('bnds,ds->bnds', h, A_mat)
            mamba_input = torch.einsum('bnd,bns->bnds', x, B_mat)
            force_1 = delta_self.unsqueeze(-1) * (mamba_decay + mamba_input)
            
            # Force 2: 2D Spatial Diffusion
            force_2 = delta_diff.unsqueeze(-1) * diffused_h
            
            # Forward Euler Integration
            h = h + dt * (force_1 + force_2)

        # 5. Output Projection: y(T) = C * h(T) + D * x
        y = torch.einsum('bnds,bns->bnd', h, C_mat)
        y = y + x * self.D
        return y

class ContinuousSpatialMambaBlock(nn.Module):
    """
    Drop-in block natively avoiding 1D sequential bias entirely.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(d_model, expand * d_model * 2, bias=False)
        
        # Native 2D Pre-Processing (Not 1D Sequence Flat!)
        self.local_conv2d = nn.Conv2d(
            in_channels=expand * d_model, out_channels=expand * d_model,
            kernel_size=3, padding=1, groups=expand * d_model, bias=True
        )
        
        self.activation = nn.SiLU()
        self.continuous_ssm = ContinuousSpatialSSM(d_model=d_model, d_state=d_state, expand=expand)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(expand * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, K_steps: int = 3) -> torch.Tensor:
        residual = x
        B_val, N, D_dim = x.shape
        H = W = int(math.sqrt(N))  # Native 2D grid shape

        x_norm = self.norm(x)
        
        xz = self.in_proj(x_norm)
        u, z = xz.chunk(2, dim=-1)   # (B, N, D_inner)
        
        # ── 2D Local Convolution (No 1D flattening sequence operations) ──
        u_2d = u.transpose(1, 2).view(B_val, -1, H, W)   # (B, D_inner, H, W)
        u_2d = self.local_conv2d(u_2d)                   # (B, D_inner, H, W)
        u = u_2d.view(B_val, -1, N).transpose(1, 2)      # (B, N, D_inner)
        
        u = self.activation(u)
        
        # ── PyTorch Memory Optimization (Gradient Checkpointing) ──
        # Fixes OOM by discarding massive intermediate loop tensors during forward pass
        if self.training:
            from torch.utils.checkpoint import checkpoint
            # use_reentrant=False is required for modern PyTorch 2.x safe checkpointing
            y_ssm = checkpoint(self.continuous_ssm, u, K_steps, use_reentrant=False)
        else:
            y_ssm = self.continuous_ssm(u, K_steps=K_steps)
            
        y = y_ssm * F.silu(z)
        y = self.out_proj(y)
        
        return y + residual

class ContinuousSpatialMambaClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from models.patch_encoder import PatchEmbedding
        self.embedder = PatchEmbedding(img_size=getattr(cfg, 'canvas_size', 128), patch_size=cfg.patch_size, in_channels=3, d_embed=cfg.d_embed)
        self.blocks = nn.ModuleList([
            ContinuousSpatialMambaBlock(cfg.d_embed, cfg.d_state)
            for _ in range(cfg.n_mamba_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_embed)
        self.head = nn.Linear(cfg.d_embed, getattr(cfg, 'n_classes', 10))
        self.K_steps = getattr(cfg, 'K_steps', 3)

    def forward(self, x):
        x = self.embedder(x)
        for block in self.blocks:
            x = block(x, K_steps=self.K_steps)
        x = self.norm(x)
        features = x.mean(dim=1)
        return self.head(features)
