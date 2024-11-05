#######################  SS

class Depth_slobal_power(nn.Module):
    def __init__(self, max_weight=5.0, epsilon=1e-8):
        super(Depth_slobal_power, self).__init__()
        self.max_weight = max_weight
        self.epsilon = epsilon

        sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                       [0, 0, 0],
                                       [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_kernel_x = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_kernel_y = nn.Parameter(sobel_kernel_y, requires_grad=False)

        laplacian_kernel = torch.tensor([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.laplacian_kernel = nn.Parameter(laplacian_kernel, requires_grad=False)

    def forward(self, depth):
        if depth.dim() == 4 and depth.size(1) == 3:

            depth = depth.mean(dim=1, keepdim=True)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(1)

        depth_normalized = depth


        depth_min = depth.view(depth.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        depth_max = depth.view(depth.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        depth_normalized = (depth - depth_min) / (depth_max - depth_min + self.epsilon)  # [B, 1, H, W]

        depth_padded = F.pad(depth_normalized, (1, 1, 1, 1), mode='reflect')  # 使用反射填充


        grad_x = F.conv2d(depth_padded, self.sobel_kernel_x.to(depth.device))
        grad_y = F.conv2d(depth_padded, self.sobel_kernel_y.to(depth.device))

        depth_grad_first_order = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.epsilon)

        depth_laplacian = F.conv2d(depth_padded, self.laplacian_kernel.to(depth.device))
        depth_grad_second_order = torch.abs(depth_laplacian)

        scales = [1, 0.75, 0.5, 0.25]
        depth_grad_multiscale = depth_grad_first_order.clone()
        depth_grad_second_multiscale = depth_grad_second_order.clone()

        for scale in scales[1:]:
            depth_scaled = F.interpolate(depth_normalized, scale_factor=scale, mode='bilinear', align_corners=True)
            depth_padded_scaled = F.pad(depth_scaled, (1, 1, 1, 1), mode='reflect')


            grad_x_scaled = F.conv2d(depth_padded_scaled, self.sobel_kernel_x.to(depth.device))
            grad_y_scaled = F.conv2d(depth_padded_scaled, self.sobel_kernel_y.to(depth.device))
            grad_first_order_scaled = torch.sqrt(grad_x_scaled ** 2 + grad_y_scaled ** 2 + self.epsilon)
            grad_first_order_scaled = F.interpolate(grad_first_order_scaled, size=depth.size()[2:], mode='bilinear', align_corners=True)
            depth_grad_multiscale += grad_first_order_scaled


            laplacian_scaled = F.conv2d(depth_padded_scaled, self.laplacian_kernel.to(depth.device))
            grad_second_order_scaled = torch.abs(laplacian_scaled)
            grad_second_order_scaled = F.interpolate(grad_second_order_scaled, size=depth.size()[2:], mode='bilinear', align_corners=True)
            depth_grad_second_multiscale += grad_second_order_scaled


        depth_grad_multiscale = depth_grad_multiscale / len(scales)
        depth_grad_second_multiscale = depth_grad_second_multiscale / len(scales)


        combined_grad = depth_grad_multiscale + depth_grad_second_multiscale


        grad_min = combined_grad.view(depth.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        grad_max = combined_grad.view(depth.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        combined_grad_normalized = (combined_grad - grad_min) / (grad_max - grad_min + self.epsilon)


        weight_map = 1 + self.max_weight * torch.tanh(combined_grad_normalized)

        weight_map = weight_map.squeeze(1)

        return weight_map

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