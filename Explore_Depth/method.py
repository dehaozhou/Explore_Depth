#######################  SS


class Depth_MoE(nn.Module):
    """重构后的 Depth MoE，用于生成深度加权图，同时输出专家门控权重"""
    def __init__(self, win=3, embed_dim=16, num_heads=2):
        super().__init__()
        self.win = win
        self.embed_dim = embed_dim
        self.depth_proj = nn.Linear(1, embed_dim)
        self.semantic_proj = nn.Linear(1, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        def make_expert():
            return nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        self.expert_geo    = make_expert()
        self.expert_sem    = make_expert()
        self.expert_fusion = make_expert()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
            nn.Softmax(dim=-1)
        )
        self.proj_out = nn.Linear(embed_dim, 1)

        # init.xavier_uniform_(self.expert_geo[0].weight)
        # init.xavier_uniform_(self.expert_sem[0].weight)
        # init.xavier_uniform_(self.expert_fusion[0].weight)

    def forward(self, depth, semantic_pred):
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        sem_prob = semantic_pred.softmax(dim=1).max(dim=1, keepdim=True).values
        B, _, H, W = depth.shape
        depth = (depth - depth.mean(dim=[2,3], keepdim=True)) \
                / (depth.std(dim=[2,3], keepdim=True) + 1e-6)
        sem_prob = (sem_prob - sem_prob.mean(dim=[2,3], keepdim=True)) \
                   / (sem_prob.std(dim=[2,3], keepdim=True) + 1e-6)
        depth = depth.detach()
        depth_patch = einops.rearrange(
            depth, 'b 1 (h w1) (w w2) -> (b h w) (w1 w2) 1',
            w1=self.win, w2=self.win
        )
        sem_patch   = einops.rearrange(
            sem_prob, 'b 1 (h w1) (w w2) -> (b h w) (w1 w2) 1',
            w1=self.win, w2=self.win
        )
        depth_feat = self.depth_proj(depth_patch)    # [N, win*win, D]
        sem_feat   = self.semantic_proj(sem_patch)   # [N, win*win, D]
        x = depth_feat + sem_feat                    # [N,win*win,D]
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)             # [N,win*win,D]
        x = self.norm1(x + attn_out)
        geo_in    = x + depth_feat
        sem_in    = x + sem_feat
        fusion_in = x + depth_feat + sem_feat
        o_geo    = geo_in    + self.expert_geo(geo_in)
        o_sem    = sem_in    + self.expert_sem(sem_in)
        o_fusion = fusion_in + self.expert_fusion(fusion_in)
        expert_outputs = torch.stack([o_geo, o_sem, o_fusion], dim=-1)
        mean_stat = x.mean(dim=1)        # [N, D]
        max_stat  = x.max(dim=1).values  # [N, D]
        std_stat  = x.std(dim=1)         # [N, D]
        stats = torch.cat([mean_stat, max_stat, std_stat], dim=-1)  # [N,3D]
        gate_w = self.gate(stats)        # [N, 3]
        avg_gate = gate_w.mean(dim=0)    # [3]
        gate_w = gate_w.unsqueeze(1).unsqueeze(2)  # [N,1,1,3]
        moe = (expert_outputs * gate_w).sum(dim=-1)  # [N, win*win, D]
        moe = self.norm2(moe + x)                    # 残差 + 归一化
        y = torch.sigmoid(self.proj_out(moe))       # [N,win*win,1]
        weight_map = einops.rearrange(
            y, '(b h w) (w1 w2) 1 -> b 1 (h w1) (w w2)',
            b=B, h=H//self.win, w=W//self.win, w1=self.win, w2=self.win
        )
        return weight_map.squeeze(1), avg_gate

class CrossEntropy2d_ignore(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d_ignore, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None, weight_map=None):
        """
        Args:
            predict: (n, c, h, w) - 预测输出
            target: (n, h, w) - 真实标签
            weight: (c,) - 类别权重
            weight_map: (n, h, w) - 像素级权重图
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        n, c, h, w = predict.size()


        target_mask = (target >= 0) & (target != self.ignore_label)
        valid_indices = target_mask.nonzero(as_tuple=True)

        if len(valid_indices[0]) == 0:
            return torch.zeros(1).cuda()


        predict = predict.permute(0, 2, 3, 1)  # (n, h, w, c)
        predict = predict[valid_indices]
        predict = predict.view(-1, c)  # (num_valid_pixels, c)
        target = target[valid_indices]  # (num_valid_pixels,)

        if weight_map is not None:
            weight_map = weight_map.to(predict.device)
            per_pixel_weight = weight_map[valid_indices]  # (num_valid_pixels,)
        else:
            per_pixel_weight = None


        loss = F.cross_entropy(predict, target, weight=weight, reduction='none')

        if per_pixel_weight is not None:
            loss = loss * per_pixel_weight

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss





#######################  UDA

class SFF(nn.Module):
    def __init__(self, dim):
        super(SFF, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
        )
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x, y):
        initial = x + y
        x_gap = self.gap(initial)
        cattn = self.ca(x_gap)
        x_avg = torch.mean(initial, dim=1, keepdim=True)
        x_max, _ = torch.max(initial, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        pattn1 = sattn + cattn
        x_initial = initial.unsqueeze(dim=2)
        pattn1 = pattn1.unsqueeze(dim=2)
        x2 = torch.cat([x_initial, pattn1], dim=2)
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result
    

##################################### FS

class DepthAwareSegmentationLoss(nn.Module):
    def __init__(self, lambda_edge=1.0, lambda_consistency=0.5):
        super(DepthAwareSegmentationLoss, self).__init__()
        self.lambda_edge = lambda_edge
        self.lambda_consistency = lambda_consistency


    def normalize_depth(self, depth):

        min_depth = torch.min(depth)
        max_depth = torch.max(depth)
        depth_normalized = (depth - min_depth) / (max_depth - min_depth + 1e-8)
        return depth_normalized

    def gradient_x(self, img):
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]
        gx = gx[:, :, :-1, :]
        return gx

    def gradient_y(self, img):
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]
        gy = gy[:, :, :, :-1]
        return gy

    def depth_edge_loss(self, pred_seg, depth):

        depth_normalized = self.normalize_depth(depth)

        # 计算深度图的梯度
        depth_grad_x = self.gradient_x(depth_normalized)
        depth_grad_y = self.gradient_y(depth_normalized)


        depth_grad_magnitude = torch.sqrt(depth_grad_x ** 2 + depth_grad_y ** 2 + 1e-8)


        pred_seg_softmax = F.softmax(pred_seg, dim=1)
        seg_boundary = self.compute_segmentation_boundaries(pred_seg_softmax)


        edge_loss = torch.mean(seg_boundary * depth_grad_magnitude)

        return edge_loss

    def compute_segmentation_boundaries(self, pred_seg_softmax):

        seg_grad_x = self.gradient_x(pred_seg_softmax)  # [B, C, H-1, W-1]
        seg_grad_y = self.gradient_y(pred_seg_softmax)  # [B, C, H-1, W-1]

        seg_grad_magnitude = torch.sqrt(seg_grad_x ** 2 + seg_grad_y ** 2 + 1e-8)

        seg_boundary = torch.sum(seg_grad_magnitude, dim=1, keepdim=True)  # [B, 1, H-1, W-1]
        return seg_boundary

    def depth_consistency_loss(self, pred_seg, depth):

        depth_normalized = self.normalize_depth(depth)


        pred_labels = torch.argmax(pred_seg, dim=1, keepdim=True)
        depth_flat = depth_normalized.view(depth_normalized.size(0), -1)
        label_flat = pred_labels.view(pred_labels.size(0), -1)

        depth_variance = 0.0
        num_classes = pred_seg.size(1)
        for c in range(num_classes):
            mask = (label_flat == c)
            if mask.sum() > 0:
                class_depths = depth_flat[mask]
                class_mean = class_depths.mean()
                class_variance = ((class_depths - class_mean) ** 2).mean()
                depth_variance += class_variance

        depth_variance = depth_variance / num_classes
        return depth_variance

    def forward(self, pred_seg,  depth):
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        depth = depth.to(pred_seg.device)
        depth = self.normalize_depth(depth)
        edge_loss = self.depth_edge_loss(pred_seg, depth)
        consistency_loss = self.depth_consistency_loss(pred_seg, depth)
        total_loss =  self.lambda_edge * edge_loss + self.lambda_consistency * consistency_loss
        return total_loss
