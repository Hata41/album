import random, math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from PIL import Image, ImageOps
from tqdm import tqdm

# =============================
# Image Metadata Caching
# =============================
@dataclass
class ImageMetadata:
    path: Path
    width: int
    height: int
    aspect_ratio: float
    pref_aspect: float

_metadata_cache: Dict[Path, ImageMetadata] = {}

def batch_process_images(paths: List[Path]) -> List[ImageMetadata]:
    results = []
    for p in tqdm(paths, desc="Processing Images"):
        if p in _metadata_cache:
            results.append(_metadata_cache[p])
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
            r = w / h if h else 1.0
            
            if 0.85 <= r <= 1.15:
                pref = 1.0
            elif r > 1.7:
                pref = 2.0
            elif r > 1.15:
                pref = 1.5
            else:
                pref = 2/3
                
            meta = ImageMetadata(p, w, h, r, pref)
            _metadata_cache[p] = meta
            results.append(meta)
        except Exception as e:
            print(f"Error processing {p}: {e}")
    return results

# =============================
# Guillotine slicing tree
# =============================
@dataclass
class Node:
    dir: Optional[str] = None  # "H" or "V"
    t: Optional[float] = None  # (0,1)
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    leaf_id: Optional[int] = None

def is_power_of_two(x: int) -> bool:
    return x >= 1 and (x & (x - 1) == 0)

def largest_power_of_two_leq(x: int) -> int:
    # assumes x >= 1
    return 1 << (x.bit_length() - 1)

def build_full_tree(num_leaves: int, seed: int = 0) -> Node:
    """
    Balanced full binary tree with num_leaves leaves.
    num_leaves must be a power of 2.
    """
    assert is_power_of_two(num_leaves), "num_leaves must be a power of 2"
    rng = random.Random(seed)
    leaves = [Node(leaf_id=i) for i in range(num_leaves)]
    current = leaves
    while len(current) > 1:
        nxt = []
        for i in range(0, len(current), 2):
            nxt.append(Node(
                dir=rng.choice(["H", "V"]),
                t=rng.uniform(0.35, 0.65),
                left=current[i],
                right=current[i + 1],
            ))
        current = nxt
    return current[0]

def internal_nodes(root: Node) -> List[Node]:
    out = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.leaf_id is None:
            out.append(n)
            stack.append(n.right)
            stack.append(n.left)
    return out

def leaf_ids(root: Node) -> List[int]:
    ids: List[int] = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.leaf_id is not None:
            ids.append(n.leaf_id)
        else:
            stack.append(n.right)
            stack.append(n.left)
    ids.sort()
    return ids

def decode_region(root: Node, x0: int, y0: int, W: int, H: int) -> Dict[int, Tuple[int, int, int, int]]:
    out: Dict[int, Tuple[int, int, int, int]] = {}

    def rec(node: Node, x: int, y: int, w: int, h: int):
        if node.leaf_id is not None:
            out[node.leaf_id] = (x, y, w, h)
            return
        t = float(node.t)
        if node.dir == "V":
            w1 = max(1, min(w - 1, int(w * t))) if w >= 2 else w
            rec(node.left, x, y, w1, h)
            rec(node.right, x + w1, y, w - w1, h)
        else:
            h1 = max(1, min(h - 1, int(h * t))) if h >= 2 else h
            rec(node.left, x, y, w, h1)
            rec(node.right, x, y + h1, w, h - h1)

    rec(root, x0, y0, W, H)
    return out

# =============================
# Energy
# =============================
def energy(root: Node, W: int, H: int, perm: List[int], prefs: np.ndarray) -> float:
    """
    prefs is a np.ndarray of target aspect ratios.
    perm[leaf_id] = image index assigned to that leaf.
    """
    boxes = decode_region(root, 0, 0, W, H)
    num_boxes = len(boxes)
    target_area = (W * H) / num_boxes

    # Vectorized part
    leaf_ids_list = sorted(boxes.keys())
    w_h = np.array([boxes[lid][2:4] for lid in leaf_ids_list]) # (num_boxes, 2)
    w = w_h[:, 0]
    h = w_h[:, 1]
    
    img_indices = np.array([perm[lid] for lid in leaf_ids_list])
    current_prefs = prefs[img_indices]
    
    rho = w / np.where(h == 0, 1.0, h)
    
    # Energy components
    e_aspect = np.sum(np.log(rho / current_prefs) ** 2)
    e_area = np.sum(((w * h - target_area) / target_area) ** 2)
    
    e = e_aspect + e_area

    # regularize extreme splits (mildly)
    for n in internal_nodes(root):
        t = min(max(float(n.t), 1e-6), 1 - 1e-6)
        e += 0.02 * (math.log(t / (1 - t)) ** 2)

    return float(e)

# =============================
# Rendering (final shrinking = gaps + page margin)
# =============================
def render_page(
    root: Node,
    page_W: int,
    page_H: int,
    images: List[Path],
    perm: List[int],
    page_margin_px: int,
    gap_px: int,
    bg=(255, 255, 255),
    title: str = "default title",
    crop_to_fit: bool = True,
) -> Image.Image:
    title_height = int(page_H * 0.1)
    inner_W = page_W - 2 * page_margin_px
    inner_H = page_H - 2 * page_margin_px - title_height
    if inner_W <= 0 or inner_H <= 0:
        raise ValueError("page_margin_px too large for the page size")

    placements = decode_region(root, page_margin_px, page_margin_px + title_height, inner_W, inner_H)

    # shrink each tile by half-gap on each side
    inset = gap_px / 2.0

    page = Image.new("RGB", (page_W, page_H), bg)
    
    # Draw title
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(page)
    try:
        font = ImageFont.truetype("arial.ttf", int(title_height * 0.4))
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", int(title_height * 0.4))
        except:
            font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (page_W - text_width) // 2
    y = (title_height - text_height) // 2
    draw.text((x, y), title, fill=(0, 0, 0), font=font)
    
    for leaf_id, (x, y, w, h) in placements.items():
        xi = int(round(x + inset))
        yi = int(round(y + inset))
        wi = int(round(w - 2 * inset))
        hi = int(round(h - 2 * inset))
        if wi <= 2 or hi <= 2:
            continue

        img_id = perm[leaf_id]
        with Image.open(images[img_id]) as im:
            im = im.convert("RGB")
            if crop_to_fit:
                tile = ImageOps.fit(im, (wi, hi), method=Image.Resampling.LANCZOS)
            else:
                tile = ImageOps.pad(im, (wi, hi), color=bg, centering=(0.5, 0.5))
            page.paste(tile, (xi, yi))

    return page

def anneal_global(
    roots: List[Node],
    page_W: int, page_H: int,
    all_images: List[Path], all_prefs: List[float],
    initial_perms: List[List[int]], # Global indices
    swap_pool: List[int],           # Global indices
    locked_leaves: List[Dict[int, int]], # List of leaf_id -> global_img_idx
    steps: int = 10000,
    progress_callback: Optional[callable] = None,
    title: str = "default title"
) -> Tuple[List[List[int]], List[int], List[float]]:
    rng = random.Random(42)
    num_pages = len(roots)
    perms = [p[:] for p in initial_perms]
    pool = swap_pool[:]
    prefs_arr = np.array(all_prefs)
    
    # Pre-calculate internal nodes and leaf IDs for each page
    pages_internal = [internal_nodes(r) for r in roots]
    pages_leaf_ids = [leaf_ids(r) for r in roots]
    
    # Pre-calculate free slots for each page (respecting locks)
    pages_free_slots = []
    for p in range(num_pages):
        locked_set = set(locked_leaves[p].keys())
        pages_free_slots.append([lid for lid in pages_leaf_ids[p] if lid not in locked_set])

    # Inner dimensions for energy calculation
    page_margin_px = 50 
    inner_W = page_W - 2 * page_margin_px
    title_height = int(page_H * 0.1)
    inner_H = page_H - 2 * page_margin_px - title_height

    # Track individual page energies
    page_energies = [energy(roots[p], inner_W, inner_H, perms[p], prefs_arr) for p in range(num_pages)]
    E = sum(page_energies)
    T = 1.5
    energy_history = []

    for step in tqdm(range(steps), desc="Global Annealing", leave=False):
        energy_history.append(E)
        if progress_callback:
            progress_callback(step, steps)

        move_type = rng.random()
        
        if move_type < 0.5:
            # Structure Move (50%)
            p_idx = rng.randrange(num_pages)
            if not pages_internal[p_idx]: continue
            n = rng.choice(pages_internal[p_idx])
            
            old_t = n.t
            old_dir = n.dir
            
            if rng.random() < 0.5:
                n.t = min(0.95, max(0.05, old_t + rng.gauss(0, 0.1 * T)))
            else:
                n.dir = "H" if old_dir == "V" else "V"
            
            new_page_E = energy(roots[p_idx], inner_W, inner_H, perms[p_idx], prefs_arr)
            E2 = E - page_energies[p_idx] + new_page_E
            
            if E2 <= E or rng.random() < math.exp((E - E2) / max(T, 1e-9)):
                E = E2
                page_energies[p_idx] = new_page_E
            else:
                n.t = old_t
                n.dir = old_dir
        else:
            # Swap Move (50%)
            swap_subcase = rng.random()
            if swap_subcase < 0.33:
                # Sub-case A: Intra-Page
                p_idx = rng.randrange(num_pages)
                free = pages_free_slots[p_idx]
                if len(free) >= 2:
                    i, j = rng.sample(free, 2)
                    perms[p_idx][i], perms[p_idx][j] = perms[p_idx][j], perms[p_idx][i]
                    
                    new_page_E = energy(roots[p_idx], inner_W, inner_H, perms[p_idx], prefs_arr)
                    E2 = E - page_energies[p_idx] + new_page_E
                    
                    if E2 <= E or rng.random() < math.exp((E - E2) / max(T, 1e-9)):
                        E = E2
                        page_energies[p_idx] = new_page_E
                    else:
                        perms[p_idx][i], perms[p_idx][j] = perms[p_idx][j], perms[p_idx][i]
            elif swap_subcase < 0.66:
                # Sub-case B: Inter-Page
                if num_pages >= 2:
                    p1, p2 = rng.sample(range(num_pages), 2)
                    free1 = pages_free_slots[p1]
                    free2 = pages_free_slots[p2]
                    if free1 and free2:
                        i = rng.choice(free1)
                        j = rng.choice(free2)
                        perms[p1][i], perms[p2][j] = perms[p2][j], perms[p1][i]
                        
                        new_E1 = energy(roots[p1], inner_W, inner_H, perms[p1], prefs_arr)
                        new_E2 = energy(roots[p2], inner_W, inner_H, perms[p2], prefs_arr)
                        E2 = E - page_energies[p1] - page_energies[p2] + new_E1 + new_E2
                        
                        if E2 <= E or rng.random() < math.exp((E - E2) / max(T, 1e-9)):
                            E = E2
                            page_energies[p1] = new_E1
                            page_energies[p2] = new_E2
                        else:
                            perms[p1][i], perms[p2][j] = perms[p2][j], perms[p1][i]
            else:
                # Sub-case C: Pool Swap
                if pool:
                    p_idx = rng.randrange(num_pages)
                    free = pages_free_slots[p_idx]
                    if free:
                        i = rng.choice(free)
                        k_idx = rng.randrange(len(pool))
                        perms[p_idx][i], pool[k_idx] = pool[k_idx], perms[p_idx][i]
                        
                        new_page_E = energy(roots[p_idx], inner_W, inner_H, perms[p_idx], prefs_arr)
                        E2 = E - page_energies[p_idx] + new_page_E
                        
                        if E2 <= E or rng.random() < math.exp((E - E2) / max(T, 1e-9)):
                            E = E2
                            page_energies[p_idx] = new_page_E
                        else:
                            perms[p_idx][i], pool[k_idx] = pool[k_idx], perms[p_idx][i]
        
        T *= 0.9994
        
    return perms, pool, energy_history

def optimize_es(
    roots: List[Node],
    page_W: int, page_H: int,
    all_images: List[Path], all_prefs: List[float],
    initial_perms: List[List[int]],
    swap_pool: List[int],
    locked_leaves: List[Dict[int, int]],
    steps: int = 10000,
    progress_callback: Optional[callable] = None,
    title: str = "default title"
) -> Tuple[List[List[int]], List[int], List[float]]:
    """
    (1+1)-Evolutionary Strategy for global layout optimization.
    """
    rng = random.Random(42)
    num_pages = len(roots)
    perms = [p[:] for p in initial_perms]
    pool = swap_pool[:]
    prefs_arr = np.array(all_prefs)
    
    pages_internal = [internal_nodes(r) for r in roots]
    pages_leaf_ids = [leaf_ids(r) for r in roots]
    pages_free_slots = []
    for p in range(num_pages):
        locked_set = set(locked_leaves[p].keys())
        pages_free_slots.append([lid for lid in pages_leaf_ids[p] if lid not in locked_set])

    page_margin_px = 50 
    inner_W = page_W - 2 * page_margin_px
    title_height = int(page_H * 0.1)
    inner_H = page_H - 2 * page_margin_px - title_height

    page_energies = [energy(roots[p], inner_W, inner_H, perms[p], prefs_arr) for p in range(num_pages)]
    E = sum(page_energies)
    
    sigma = 0.1
    energy_history = []
    
    success_window = []
    window_size = 50

    for step in tqdm(range(steps), desc="Evolutionary Strategy", leave=False):
        energy_history.append(E)
        if progress_callback:
            progress_callback(step, steps)

        p_idx = rng.randrange(num_pages)
        mutation_accepted = False
        
        move_type = rng.random()
        
        if move_type < 0.9: # 90% Gaussian mutation on t
            if pages_internal[p_idx]:
                n = rng.choice(pages_internal[p_idx])
                old_t = n.t
                n.t = min(0.95, max(0.05, old_t + rng.gauss(0, sigma)))
                
                new_page_E = energy(roots[p_idx], inner_W, inner_H, perms[p_idx], prefs_arr)
                E2 = E - page_energies[p_idx] + new_page_E
                
                if E2 < E:
                    E = E2
                    page_energies[p_idx] = new_page_E
                    mutation_accepted = True
                else:
                    n.t = old_t
        else: # 10% Discrete mutation (Flip or Swap)
            sub_move = rng.random()
            if sub_move < 0.33: # Flip H/V
                if pages_internal[p_idx]:
                    n = rng.choice(pages_internal[p_idx])
                    old_dir = n.dir
                    n.dir = "H" if old_dir == "V" else "V"
                    
                    new_page_E = energy(roots[p_idx], inner_W, inner_H, perms[p_idx], prefs_arr)
                    E2 = E - page_energies[p_idx] + new_page_E
                    
                    if E2 < E:
                        E = E2
                        page_energies[p_idx] = new_page_E
                        mutation_accepted = True
                    else:
                        n.dir = old_dir
            elif sub_move < 0.66: # Intra-page swap
                free = pages_free_slots[p_idx]
                if len(free) >= 2:
                    i, j = rng.sample(free, 2)
                    perms[p_idx][i], perms[p_idx][j] = perms[p_idx][j], perms[p_idx][i]
                    
                    new_page_E = energy(roots[p_idx], inner_W, inner_H, perms[p_idx], prefs_arr)
                    E2 = E - page_energies[p_idx] + new_page_E
                    
                    if E2 < E:
                        E = E2
                        page_energies[p_idx] = new_page_E
                        mutation_accepted = True
                    else:
                        perms[p_idx][i], perms[p_idx][j] = perms[p_idx][j], perms[p_idx][i]
            else: # Pool swap
                if pool:
                    free = pages_free_slots[p_idx]
                    if free:
                        i = rng.choice(free)
                        k_idx = rng.randrange(len(pool))
                        perms[p_idx][i], pool[k_idx] = pool[k_idx], perms[p_idx][i]
                        
                        new_page_E = energy(roots[p_idx], inner_W, inner_H, perms[p_idx], prefs_arr)
                        E2 = E - page_energies[p_idx] + new_page_E
                        
                        if E2 < E:
                            E = E2
                            page_energies[p_idx] = new_page_E
                            mutation_accepted = True
                        else:
                            perms[p_idx][i], pool[k_idx] = pool[k_idx], perms[p_idx][i]

        success_window.append(mutation_accepted)
        if len(success_window) >= window_size:
            success_rate = sum(success_window) / window_size
            if success_rate > 0.2:
                sigma *= 1.2
            else:
                sigma *= 0.85
            success_window = []

    return perms, pool, energy_history

def optimize_linear_partition(
    roots: List[Node],
    page_W: int, page_H: int,
    all_images: List[Path], all_prefs: List[float],
    initial_perms: List[List[int]],
    swap_pool: List[int],
    locked_leaves: List[Dict[int, int]],
    steps: int = 1,
    progress_callback: Optional[callable] = None,
    title: str = "default title"
) -> Tuple[List[Node], List[List[int]], List[int], List[float]]:
    """
    Deterministic Linear Partitioning (Justified Layout) algorithm.
    Arranges images in sequential order into rows, minimizing cropping.
    """
    num_pages = len(roots)
    new_roots = []
    perms = [p[:] for p in initial_perms]
    pool = swap_pool[:]
    
    page_margin_px = 50 
    title_height = int(page_H * 0.1)
    inner_W = page_W - 2 * page_margin_px
    inner_H = page_H - 2 * page_margin_px - title_height

    for p_idx in range(num_pages):
        perm = perms[p_idx]
        if not perm:
            new_roots.append(roots[p_idx])
            continue
            
        prefs = [all_prefs[idx] for idx in perm]
        num_images = len(perm)
        
        # Target row height: heuristic to have ~3 images per row
        target_h = inner_H / math.sqrt(num_images) if num_images > 0 else inner_H
        
        # DP for optimal row breaks
        # dp[i] = min cost to partition first i images
        dp = [float('inf')] * (num_images + 1)
        dp[0] = 0
        breaks = [0] * (num_images + 1)
        
        for i in range(1, num_images + 1):
            # Iterate backwards to find the best previous break
            # Limit images per row to 5 for better aesthetics
            for j in range(max(0, i - 5), i):
                row_prefs = prefs[j:i]
                row_width = sum(r * target_h for r in row_prefs)
                # Cost is squared difference from inner_W
                cost = dp[j] + (row_width - inner_W)**2
                if cost < dp[i]:
                    dp[i] = cost
                    breaks[i] = j
        
        # Reconstruct rows
        rows_indices = []
        curr = num_images
        while curr > 0:
            prev = breaks[curr]
            rows_indices.append(list(range(prev, curr)))
            curr = prev
        rows_indices.reverse()
        
        # Build Tree from rows
        def build_row_tree(indices):
            if len(indices) == 1:
                return Node(leaf_id=indices[0])
            
            mid = len(indices) // 2
            left_indices = indices[:mid]
            right_indices = indices[mid:]
            
            left_sum = sum(prefs[idx] for idx in left_indices)
            right_sum = sum(prefs[idx] for idx in right_indices)
            t = left_sum / (left_sum + right_sum) if (left_sum + right_sum) > 0 else 0.5
            
            return Node(
                dir="V",
                t=t,
                left=build_row_tree(left_indices),
                right=build_row_tree(right_indices)
            )

        def build_page_tree(row_list):
            if len(row_list) == 1:
                return build_row_tree(row_list[0])
            
            mid = len(row_list) // 2
            left_rows = row_list[:mid]
            right_rows = row_list[mid:]
            
            def get_rows_height_sum(r_list):
                h_sum = 0
                for r in r_list:
                    s = sum(prefs[idx] for idx in r)
                    h_sum += inner_W / s if s > 0 else target_h
                return h_sum
            
            h_left = get_rows_height_sum(left_rows)
            h_right = get_rows_height_sum(right_rows)
            t = h_left / (h_left + h_right) if (h_left + h_right) > 0 else 0.5
            
            return Node(
                dir="H",
                t=t,
                left=build_page_tree(left_rows),
                right=build_page_tree(right_rows)
            )
            
        new_roots.append(build_page_tree(rows_indices))
        
    if progress_callback:
        progress_callback(100, 100)
        
    return new_roots, perms, pool, [0.0]

# =============================
# Aspect preference bucketing (replace if you have explicit m formats)
# =============================
def pref_aspect_for(path: Path) -> float:
    if path in _metadata_cache:
        return _metadata_cache[path].pref_aspect
    with Image.open(path) as im:
        w, h = im.size
    r = w / h if h else 1.0
    if 0.85 <= r <= 1.15:
        return 1.0
    if r > 1.7:
        return 2.0
    if r > 1.15:
        return 1.5
    return 2/3
