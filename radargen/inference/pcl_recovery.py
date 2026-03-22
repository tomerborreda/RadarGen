import numpy as np
import torch
from typing import Optional


def create_matched_kernel_torch(
    sigma_x: float, 
    sigma_y: float, 
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    sigma_threshold_sq = 3.0**2
    
    # 1. Determine kernel size from 3-sigma truncation
    size_x = int(3.0 * sigma_x * 2 + 1)
    size_y = int(3.0 * sigma_y * 2 + 1)
    if size_x % 2 == 0: size_x += 1
    if size_y % 2 == 0: size_y += 1
    
    half_size_x = size_x // 2
    half_size_y = size_y // 2
    
    # 2. Create grids, using 'xy' indexing to match
    x, y = torch.meshgrid(
        torch.arange(-half_size_x, half_size_x + 1, device=device),
        torch.arange(-half_size_y, half_size_y + 1, device=device),
        indexing='xy' 
    )
    
    # 3. Calculate exponent
    exponent = -0.5 * (x**2 / sigma_x**2 + y**2 / sigma_y**2)
    
    # 4. Apply exponent
    g = torch.exp(exponent)
    
    # 5. Apply 3-sigma truncation
    mask = (x**2 / sigma_x**2 + y**2 / sigma_y**2) < sigma_threshold_sq
    g = g * mask
    
    # 6. Apply coefficient from the user's function
    coeff = 1.0 / (2.0 * np.pi * sigma_x * sigma_y)
    kernel = coeff * g

    # Return kernel and its padding size
    return kernel, (half_size_y, half_size_x)


def soft_threshold(x, thresh):
    """Soft-thresholding operator (proximal for L1)."""
    return torch.sign(x) * torch.relu(torch.abs(x) - thresh)


def solve_weighted_fista(
        y_img: torch.Tensor, 
        kernel_fft: torch.Tensor, 
        kernel_fft_conj: torch.Tensor, 
        L: float, 
        weights: torch.Tensor, 
        alpha: float, 
        n_fista_iter: int, 
        tol: float,
    ) -> torch.Tensor:
    """
    Solves the weighted Lasso problem. This is the inner loop for IRL1.
    """
    # --- 1. Initialize FISTA variables ---
    x_k = torch.zeros_like(y_img)
    t_k = 1.0
    y_k = x_k.clone()
    x_prev = x_k.clone()
    
    # --- 2. Pre-calculate the 2D threshold map ---
    threshold_map = (alpha * weights) / L

    # --- 3. Run FISTA Iteration Loop ---
    for i in range(n_fista_iter):
        # --- Gradient step ---
        y_k_fft = torch.fft.fft2(y_k)
        Ay = torch.fft.ifft2(kernel_fft * y_k_fft).real
        
        residual = Ay - y_img
        residual_fft = torch.fft.fft2(residual)
        grad = torch.fft.ifft2(kernel_fft_conj * residual_fft).real
        
        # --- Proximal (shrinkage) step ---
        x_k_next = soft_threshold(y_k - grad / L, threshold_map)
        
        # --- Nesterov Acceleration ---
        t_k_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0
        y_k = x_k_next + ((t_k - 1.0) / t_k_next) * (x_k_next - x_k)
        
        x_k = x_k_next
        t_k = t_k_next

        # --- Check for convergence ---
        if i % 20 == 0:
            change = torch.norm(x_k - x_prev) / (torch.norm(x_prev) + 1e-9)
            if i > 10 and change < tol:
                break
            x_prev = x_k.clone()
            
    return x_k


def solve_irl1(
        y_img: torch.Tensor, 
        sigma_x: float, 
        sigma_y: float, 
        alpha: float, 
        n_irl1_iter: int, 
        n_fista_iter: int, 
        epsilon: float = 1e-6, 
        tol: float = 1e-5):
    """
    Solves the sparse deconvolution problem using IRL1 in the Fourier domain.
    This function creates the FFT kernel from the sigma values.
    
    Args:
        y_img (Tensor [H, W]): The target image.
        sigma_x (float): Sigma for the kernel (x-axis).
        sigma_y (float): Sigma for the kernel (y-axis).
        alpha (float): The L1 regularization strength.
        n_irl1_iter (int): Number of outer IRL1 re-weighting iterations.
        n_fista_iter (int): Number of inner FISTA iterations per re-weighting.
        epsilon (float): Stability constant for weight calculation (1 / (|x| + eps)).
        tol (float): Convergence tolerance for inner FISTA loop.
    """
    H, W = y_img.shape
    device = y_img.device
    
    # --- 1. Create the FFT Kernel (Done once) ---
    kernel, _ = create_matched_kernel_torch(sigma_x, sigma_y, device=device)
    kH, kW = kernel.shape
    psf = torch.zeros((H, W), device=device)
    cy, cx = H // 2, W // 2
    top, left = cy - kH // 2, cx - kW // 2
    bottom, right = top + kH, left + kW
    psf[top:bottom, left:right] = kernel
    kernel_fft = torch.fft.fft2(torch.fft.ifftshift(psf))

    # --- 2. Pre-compute L and Conjugate Kernel (Done once) ---
    with torch.no_grad():
        L = torch.max(torch.abs(kernel_fft)**2).item() * 1.05
    kernel_fft_conj = torch.conj(kernel_fft)
    
    # --- 3. Initialize IRL1 variables ---
    weights = torch.ones((H, W), device=device)
    x_solved = torch.zeros((H, W), device=device)

    # --- 4. Run the IRL1 Outer Loop ---
    for _ in range(n_irl1_iter):
        # 4a. Solve the weighted L1 problem using the fast FISTA inner loop
        x_solved = solve_weighted_fista(
            y_img,
            kernel_fft,
            kernel_fft_conj,
            L,
            weights,
            alpha,
            n_fista_iter,
            tol
        )
        
        # 4b. Update weights for the next iteration
        weights = 1.0 / (torch.abs(x_solved) + epsilon)
    
    return x_solved


def deconv_cell_recovery(
    pd_map: np.ndarray,
    sigma_pix: float = 2.0,
    lam: float = 0.0018,
    threshold: float = 0.1,
    n_fista_iter: int = 300,
    n_irl1_iter: int = 5,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Extract cell indices from a point density map using deconvolution.

    Returns:
        Array of (y, x) indices of detected cells
    """
    # Auto-detect device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Normalize the map to have reasonable values for deconvolution
    if pd_map.max() <= 0:
        raise ValueError('Point density map is invalid')
    pd_map_normalized = pd_map / pd_map.max()

    if pd_map_normalized.ndim != 2:
        raise ValueError("Input image must be a 2D NumPy array.")
    y = torch.from_numpy(pd_map_normalized).float().to(device)
    recovered_map = solve_irl1(y, sigma_pix, sigma_pix, alpha=lam, n_irl1_iter=n_irl1_iter, n_fista_iter=n_fista_iter)
    cell_indices = torch.nonzero(recovered_map > threshold)
    if cell_indices.numel() == 0:
        return np.empty((0, 2), dtype=int)
    return cell_indices.cpu().numpy()
    

def recover_pcl_attributes(
        pd_map: np.ndarray,
        rcs_map: np.ndarray,
        doppler_map: np.ndarray,
        image_size: int,
        coordinate_range: float,
        sigma_pix: float = 2.0,
        lam: float = 0.0018,
        threshold: float = 0.1,
        n_fista_iter: int = 300,
        n_irl1_iter: int = 5,
        device: Optional[torch.device] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        recovered_indices = deconv_cell_recovery(
            pd_map, sigma_pix=sigma_pix, lam=lam, threshold=threshold, n_fista_iter=n_fista_iter, n_irl1_iter=n_irl1_iter, device=device)
        recovered_indices = recovered_indices[:, ::-1]

        # Sample RCS values at the sampled locations
        recovered_rcs_values = np.array([
            rcs_map[y, x] for x, y in recovered_indices
        ])
        recovered_rcs_values = np.rint(recovered_rcs_values)
        # Sample Doppler values at the sampled locations
        recovered_doppler_values = np.array([
            doppler_map[y, x] for x, y in recovered_indices
        ])

        recovered_locations = recovered_indices.copy()
        # Flip y-axis to match a coordinate system where the origin is at the bottom left corner
        recovered_locations[:, 1] = image_size - (recovered_locations[:, 1] + 1)
        # Set the location as the center of the pixel/voxel
        recovered_locations = recovered_locations.astype(np.float32) + 0.5
        # Convert the locations to the metric range
        recovered_locations = ((recovered_locations / image_size) * (2 * coordinate_range)) - coordinate_range

        return recovered_locations, recovered_rcs_values, recovered_doppler_values
