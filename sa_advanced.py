import random, math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from PIL import Image, ImageOps
from tqdm import tqdm

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
def energy(root: Node, W: int, H: int, perm: List[int], prefs: List[float]) -> float:
    """
    prefs[i] is the target aspect ratio bucket for image i in the global pool.
    perm[leaf_id] = image index assigned to that leaf.
    """
    boxes = decode_region(root, 0, 0, W, H)
    target_area = (W * H) / len(boxes)

    e = 0.0
    for leaf_id, (_, _, w, h) in boxes.items():
        rho = w / h if h else 1.0
        a = prefs[perm[leaf_id]]
        e += (math.log(rho / a) ** 2)
        e += ((w * h - target_area) / target_area) ** 2

    # regularize extreme splits (mildly)
    for n in internal_nodes(root):
        t = min(max(float(n.t), 1e-6), 1 - 1e-6)
        e += 0.02 * (math.log(t / (1 - t)) ** 2)

    return e

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
            tile = ImageOps.fit(im, (wi, hi), method=Image.Resampling.LANCZOS)
            page.paste(tile, (xi, yi))

    return page

# =============================
# SA with tqdm + snapshots only
# =============================
def anneal_with_snapshots(
    root: Node,
    page_W: int,
    page_H: int,
    all_images: List[Path],
    all_prefs: List[float],
    page_margin_px: int,
    gap_px: int,
    steps: int,
    snapshots_count: int,
    seed: int,
    desc: str,
    locked_leaves: Optional[Dict[int, int]] = None,  # leaf_id -> image_idx
    progress_callback: Optional[callable] = None,    # fn(step, total_steps)
    title: str = "default title",
):
    rng = random.Random(seed)
    nodes = internal_nodes(root)
    pool_size = len(all_prefs)

    if locked_leaves is None:
        locked_leaves = {}

    leaf_id_list = leaf_ids(root)
    leaf_count = len(leaf_id_list)

    if pool_size < leaf_count:
        raise ValueError("Not enough images to fill all leaves")

    # Initialize perm
    # 1. Place locked images
    # 2. Shuffle remaining images into remaining slots
    perm = [-1] * leaf_count
    used_images = set()

    # Apply locks
    valid_locked = {}
    for leaf_id, img_idx in locked_leaves.items():
        if leaf_id in leaf_id_list and 0 <= img_idx < pool_size:
            if img_idx in used_images:
                continue  # prevent duplicate usage of the same image in different locked leaves
            perm[leaf_id] = img_idx
            used_images.add(img_idx)
            valid_locked[leaf_id] = img_idx

    locked_leaves = valid_locked
    locked_set = set(locked_leaves.keys())

    unused_images = set(range(pool_size)) - used_images
    # Fill rest
    remaining_images = list(unused_images)
    rng.shuffle(remaining_images)
    
    for leaf_id in leaf_id_list:
        if perm[leaf_id] == -1:
            perm[leaf_id] = remaining_images.pop()
            used_images.add(perm[leaf_id])
            unused_images.discard(perm[leaf_id])
            
    # We only swap indices that are NOT locked.
    # free_slots contains all leaf indices that are allowed to be swapped.
    free_slots = [lid for lid in leaf_id_list if lid not in locked_set]
    
    inner_W = page_W - 2 * page_margin_px
    title_height = int(page_H * 0.1)
    inner_H = page_H - 2 * page_margin_px - title_height

    snapshot_steps = sorted({
        int(i * (steps - 1) / max(1, snapshots_count - 1))
        for i in range(snapshots_count)
    })
    snapshots: List[Image.Image] = []

    E = energy(root, inner_W, inner_H, perm, all_prefs)
    T = 1.5
    energy_history = []

    for step in tqdm(range(steps), desc=desc, leave=False):
        energy_history.append(E)
        if progress_callback:
            progress_callback(step, steps)

        if step in snapshot_steps:
            snapshots.append(render_page(
                root, page_W, page_H, all_images, perm,
                page_margin_px, gap_px, title=title
            ))

        move = rng.random()

        if move < 0.60:
            n = rng.choice(nodes)
            old = float(n.t)
            n.t = min(0.95, max(0.05, old + rng.gauss(0, 0.1 * T)))
            E2 = energy(root, inner_W, inner_H, perm, all_prefs)
            if E2 <= E or rng.random() < math.exp((E - E2) / max(T, 1e-9)):
                E = E2
            else:
                n.t = old

        elif move < 0.80:
            n = rng.choice(nodes)
            old = n.dir
            n.dir = "H" if old == "V" else "V"
            E2 = energy(root, inner_W, inner_H, perm, all_prefs)
            if E2 <= E or rng.random() < math.exp((E - E2) / max(T, 1e-9)):
                E = E2
            else:
                n.dir = old

        else:
            # Swap two page images, or swap a page image with an unused pool image.
            if free_slots:
                use_pool_swap = unused_images and pool_size > leaf_count and rng.random() < 0.5

                if use_pool_swap:
                    i = rng.choice(free_slots)
                    k = rng.choice(tuple(unused_images))
                    old_img = perm[i]

                    perm[i] = k
                    used_images.add(k)
                    unused_images.remove(k)
                    used_images.discard(old_img)
                    unused_images.add(old_img)

                    E2 = energy(root, inner_W, inner_H, perm, all_prefs)
                    if E2 <= E or rng.random() < math.exp((E - E2) / max(T, 1e-9)):
                        E = E2
                    else:
                        perm[i] = old_img
                        used_images.add(old_img)
                        unused_images.remove(old_img)
                        used_images.discard(k)
                        unused_images.add(k)

                elif len(free_slots) >= 2:
                    i, j = rng.sample(free_slots, 2)
                    perm[i], perm[j] = perm[j], perm[i]
                    E2 = energy(root, inner_W, inner_H, perm, all_prefs)
                    if E2 <= E or rng.random() < math.exp((E - E2) / max(T, 1e-9)):
                        E = E2
                    else:
                        perm[i], perm[j] = perm[j], perm[i]

        T *= 0.9994

    if (steps - 1) not in snapshot_steps:
        snapshots.append(render_page(
            root, page_W, page_H, all_images, perm,
            page_margin_px, gap_px, title=title
        ))

    return perm, snapshots, energy_history

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
    page_energies = [energy(roots[p], inner_W, inner_H, perms[p], all_prefs) for p in range(num_pages)]
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
            
            new_page_E = energy(roots[p_idx], inner_W, inner_H, perms[p_idx], all_prefs)
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
                    
                    new_page_E = energy(roots[p_idx], inner_W, inner_H, perms[p_idx], all_prefs)
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
                        
                        new_E1 = energy(roots[p1], inner_W, inner_H, perms[p1], all_prefs)
                        new_E2 = energy(roots[p2], inner_W, inner_H, perms[p2], all_prefs)
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
                        
                        new_page_E = energy(roots[p_idx], inner_W, inner_H, perms[p_idx], all_prefs)
                        E2 = E - page_energies[p_idx] + new_page_E
                        
                        if E2 <= E or rng.random() < math.exp((E - E2) / max(T, 1e-9)):
                            E = E2
                            page_energies[p_idx] = new_page_E
                        else:
                            perms[p_idx][i], pool[k_idx] = pool[k_idx], perms[p_idx][i]
        
        T *= 0.9994
        
    return perms, pool, energy_history

# =============================
# Storyboard (snapshots only)
# =============================
def compose_storyboard(snapshots: List[Image.Image], pad: int = 24, bg=(255, 255, 255)) -> Image.Image:
    W = sum(im.width for im in snapshots) + pad * (len(snapshots) + 1)
    H = max(im.height for im in snapshots) + pad * 2
    canvas = Image.new("RGB", (W, H), bg)
    x = pad
    for im in snapshots:
        canvas.paste(im, (x, pad))
        x += im.width + pad
    return canvas

# =============================
# Paging without duplicates:
# Use full pages of max_leaves (power of 2),
# then for the tail use decreasing powers of 2 (no repeats)
# =============================
def split_into_pages_no_duplicates(paths: List[Path], max_leaves: int) -> List[List[Path]]:
    """
    Returns a list of pages, each page length is a power of 2 <= max_leaves.
    Covers all images exactly once. No duplicates.
    """
    assert is_power_of_two(max_leaves), "max_leaves must be a power of 2"
    pages: List[List[Path]] = []
    i = 0
    n = len(paths)
    while i < n:
        remaining = n - i
        k = max_leaves if remaining >= max_leaves else largest_power_of_two_leq(remaining)
        pages.append(paths[i:i + k])
        i += k
    return pages

# =============================
# Aspect preference bucketing (replace if you have explicit m formats)
# =============================
def pref_aspect_for(path: Path) -> float:
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

# =============================
# Multi-page driver
# =============================
def make_multi_page_storyboards_no_duplicates(
    input_dir: str,
    output_dir: str,
    max_leaves_per_page: int = 16,                 # power of 2
    page_size_px: Tuple[int, int] = (2480, 3508),  # A4 portrait @ 300dpi
    page_margin_px: int = 120,
    gap_px: int = 40,
    steps: int = 15000,
    snapshots_count: int = 6,
    seed: int = 42,
):
    if not is_power_of_two(max_leaves_per_page):
        raise ValueError("max_leaves_per_page must be a power of 2 (8, 16, 32, ...)")

    input_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    all_imgs = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]
    all_imgs.sort()  # chronological if filenames encode time/order
    if not all_imgs:
        raise ValueError(f"No images found in: {input_dir}")

    pages = split_into_pages_no_duplicates(all_imgs, max_leaves_per_page)

    page_W, page_H = page_size_px
    outputs = []

    for page_idx, chunk in enumerate(tqdm(pages, desc="Pages"), start=1):
        L = len(chunk)  # power of 2 by construction
        prefs = [pref_aspect_for(p) for p in chunk]

        root = build_full_tree(L, seed=seed + page_idx)

        perm, snaps, _ = anneal_with_snapshots(
            root=root,
            page_W=page_W,
            page_H=page_H,
            all_images=chunk,
            all_prefs=prefs,
            page_margin_px=page_margin_px,
            gap_px=gap_px,
            steps=steps,
            snapshots_count=snapshots_count,
            seed=seed + 1000 + page_idx,
            desc=f"Annealing p{page_idx:03d} (L={L})",
        )

        storyboard = compose_storyboard(snaps, pad=24)
        storyboard_path = out_dir / f"page_{page_idx:03d}_storyboard.png"
        storyboard.save(storyboard_path)

        final_page = render_page(
            root=root,
            page_W=page_W,
            page_H=page_H,
            images=chunk,
            perm=perm,
            page_margin_px=page_margin_px,
            gap_px=gap_px,
        )
        final_path = out_dir / f"page_{page_idx:03d}_final.png"
        final_page.save(final_path)

        outputs.append((storyboard_path, final_path, L))

    return outputs

if __name__ == "__main__":
    # Put your photos into ./photos
    # Output:
    #  - page_XXX_storyboard.png (snapshots only)
    #  - page_XXX_final.png      (final layout)
    #
    # No duplicates: the tail is split into smaller power-of-two pages.
    outputs = make_multi_page_storyboards_no_duplicates(
        input_dir="/home/sreboul/Workspace/nothing/Test photos/2024-06-08",
        output_dir="album_portrait_storyboards",
        max_leaves_per_page=16,
        page_size_px=(2480, 3508),  # A4 portrait @ 300 DPI
        page_margin_px=120,
        gap_px=40,
        steps=15000,
        snapshots_count=6,
        seed=42,
    )

    for storyboard_path, final_path, L in outputs:
        print(f"L={L}  storyboard={storyboard_path}  final={final_path}")
