```markdown
# Esthetic Album Designer

**Esthetic Album Designer** is an intelligent photo layout generation tool that uses **Simulated Annealing** to automatically arrange images into aesthetically pleasing collages. 

Unlike standard grid layouts, this tool optimizes the size and position of every photo slot to match the aspect ratio of your specific images, minimizing awkward cropping while maintaining a perfectly aligned "guillotine" cut structure. It includes a GUI for interactive design and specific image placement.

## 🌟 Features

*   **Automatic Layout Optimization:** Uses a physics-inspired algorithm to find the best layout for a set of images.
*   **Aspect Ratio Matching:** intelligently pairs landscape slots with landscape images (and portrait with portrait) to preserve image content.
*   **Interactive GUI:** 
    *   Drag-and-drop images to specific slots to "lock" them in place.
    *   Visualize the optimization process.
*   **Guillotine Slicing Structure:** Generates layouts that can be physically cut using a paper cutter (recursive horizontal/vertical splits).
*   **High-Resolution Rendering:** Outputs print-ready images (e.g., A4 at 300 DPI).

## 🛠️ Installation

### Prerequisites
*   Python 3.10+

### Setup
1.  **Clone or Download** the repository.
2.  **Install Dependencies**:
    ```bash
    pip install PyQt6 Pillow tqdm
    ```

## 🚀 Usage

### 1. Graphical User Interface (Recommended)
The GUI allows you to load a folder, visualize the layout, and manually constrain specific photos to specific positions before optimizing.

```bash
python album_gui.py
```

*   **Load Folder:** Select a directory containing images (JPG, PNG, WEBP).
*   **Locking:** Drag an image from the left sidebar onto a specific slot in the layout. This "locks" that image to that specific rectangle.
*   **Optimize:** Click "✨ Optimize Layout" to run the algorithm. The layout will shift and resize to accommodate the aspect ratios of the images, respecting your locks.

### 2. Command Line / Batch Processing
You can run the advanced script directly to generate multi-page storyboards from a large folder of images without user intervention.

Open `sa_advanced.py` and modify the parameters at the bottom of the file (in the `if __name__ == "__main__":` block), then run:

```bash
python sa_advanced.py
```

## 🧠 How It Works: The Algorithm

The core of this project relies on **Simulated Annealing (SA)** operating on a **Guillotine Slicing Tree**.

### 1. The Data Structure: Guillotine Tree
The layout is represented as a binary tree where:
*   **Internal Nodes** represent a cut. They store:
    *   `dir`: Direction of cut (Horizontal or Vertical).
    *   `t`: The split ratio (e.g., 0.5 cuts strictly in half, 0.3 cuts at 30%).
*   **Leaf Nodes** represent the actual photo slots.

This ensures that the resulting layout is always rectangular and topologically valid (no overlapping images).

### 2. The Optimization Process (Simulated Annealing)
The algorithm starts with a random layout and iteratively attempts to improve it. It mimics the process of annealing in metallurgy: heating a material and slowly cooling it to remove defects.

#### The Steps:
1.  **Initialization:** Create a random binary tree and assign images to leaves randomly (respecting any user-defined locks).
2.  **Perturbation (The "Move"):** At every step, the algorithm makes a small random change:
    *   *Shift:* Change the split ratio `t` of a cut (move a border).
    *   *Flip:* Change a cut from Horizontal to Vertical or vice versa.
    *   *Swap:* Exchange the positions of two unlocked images.
3.  **Energy Calculation:** Calculate how "bad" the new layout is (see math below).
4.  **Acceptance Criterion:**
    *   If the new Energy is lower (better), accept the change.
    *   If the new Energy is higher (worse), accept it with probability $P = e^{-\Delta E / T}$.
    *   *Note:* This allows the algorithm to escape local optima by occasionally accepting bad moves early on.
5.  **Cooling:** Decrease the Temperature $T$ slightly ($T_{new} = T_{old} \times 0.9994$). As $T$ approaches 0, the algorithm stops accepting bad moves and settles into a final solution.

### 3. The Mathematics (Energy Function)
The "Energy" ($E$) defines what makes a layout look good. It is calculated as:

$$ E = \sum_{leaves} (E_{shape} + E_{area}) + E_{regularization} $$

#### A. Shape Error ($E_{shape}$)
This ensures the slot shape matches the image content to minimize cropping.
$$ E_{shape} = (\ln(\frac{\rho}{a}))^2 $$
*   $\rho$: The aspect ratio of the slot (width/height).
*   $a$: The preferred aspect ratio of the image inside that slot.
*   *Why Log?* Using a natural log ensures that a $2:1$ mismatch is penalized exactly the same as a $1:2$ mismatch.

#### B. Area Error ($E_{area}$)
This ensures all images get roughly equal screen space.
$$ E_{area} = (\frac{area_{slot} - area_{target}}{area_{target}})^2 $$

#### C. Regularization
This prevents the algorithm from creating impossibly thin slivers (e.g., a cut at 1% width).
$$ E_{reg} = 0.02 \times (\ln(\frac{t}{1-t}))^2 $$
This term shoots to infinity as the cut ratio $t$ approaches 0 or 1, forcing cuts to stay central.

## 📂 File Structure

*   **`album_gui.py`**: The PyQt6 application entry point. Handles UI, drag-and-drop logic, and threads the optimization.
*   **`sa_advanced.py`**: The backend logic. Contains the Tree class, the Energy function, the Simulated Annealing loop, and image rendering code.
*   **`pyproject.toml`**: Project metadata and dependencies.
```