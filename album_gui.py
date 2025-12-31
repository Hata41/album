import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, 
                             QGraphicsView, QGraphicsScene, QGraphicsRectItem, 
                             QGraphicsPixmapItem, QFileDialog, QLabel, QProgressBar,
                             QSplitter, QMessageBox, QFrame, QComboBox, QSpinBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QMimeData, QPointF
from PyQt6.QtGui import QPixmap, QDrag, QImage, QPainter, QColor, QPen, QIcon

import sa_advanced as sa
from PIL import Image, ImageOps

class OptimizationThread(QThread):
    progress = pyqtSignal(int, int)
    finished_optim = pyqtSignal(list)
    
    def __init__(self, chunks, swap_pool, pages_locks, all_prefs, image_paths, page_W, page_H):
        super().__init__()
        self.chunks = chunks
        self.swap_pool = swap_pool.copy()
        self.pages_locks = pages_locks
        self.all_prefs = all_prefs
        self.image_paths = image_paths
        self.page_W = page_W
        self.page_H = page_H
        
    def run(self):
        results = []
        for page_idx, chunk in enumerate(self.chunks):
            current_all_images = [self.image_paths[i] for i in chunk + self.swap_pool]
            current_all_prefs = [self.all_prefs[i] for i in chunk + self.swap_pool]
            
            # Map locked to local indices
            locked = {}
            for leaf_id, global_img_idx in self.pages_locks[page_idx].items():
                if global_img_idx in chunk + self.swap_pool:
                    local_idx = (chunk + self.swap_pool).index(global_img_idx)
                    locked[leaf_id] = local_idx
            
            root = sa.build_full_tree(len(chunk), seed=42 + page_idx)
            
            perm, _ = sa.anneal_with_snapshots(
                root=root,
                page_W=self.page_W,
                page_H=self.page_H,
                all_images=current_all_images,
                all_prefs=current_all_prefs,
                page_margin_px=50,
                gap_px=20,
                steps=5000,
                snapshots_count=0,
                seed=42 + page_idx,
                desc=f"Optimizing Page {page_idx + 1}",
                locked_leaves=locked,
                progress_callback=self.emit_progress
            )
            # perm is local indices, map back to global indices
            local_to_global = chunk + self.swap_pool
            global_perm = [local_to_global[local_idx] for local_idx in perm]
            results.append(global_perm)
            
            # Remove used images from swap_pool
            used = set(global_perm)
            self.swap_pool = [img for img in self.swap_pool if img not in used]
        
        self.finished_optim.emit(results)
        
    def emit_progress(self, step, total):
        self.progress.emit(step, total)

class LeafItem(QGraphicsRectItem):
    def __init__(self, x, y, w, h, leaf_id, parent_gui):
        super().__init__(x, y, w, h)
        self.leaf_id = leaf_id
        self.parent_gui = parent_gui
        self.setAcceptDrops(True)
        self.setPen(QPen(Qt.GlobalColor.black))
        self.setBrush(QColor(240, 240, 240))
        
        self.pixmap_item = QGraphicsPixmapItem(self)
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        
        # Lock icon (simple red border or overlay for now)
        self.is_locked = False
        
    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)
        if self.is_locked:
            painter.setPen(QPen(Qt.GlobalColor.red, 4))
            painter.drawRect(self.rect())
            
    def dropEvent(self, event):
        if event.mimeData().hasFormat("application/x-image-idx"):
            data = event.mimeData().data("application/x-image-idx")
            img_idx = int(data.data().decode())
            self.parent_gui.handle_drop(self.leaf_id, img_idx)
            event.accept()
        else:
            event.ignore()

class AlbumWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Esthetic Album Designer")
        self.resize(1200, 800)
        
        # State
        self.image_paths: List[Path] = []
        self.pages_roots: List[Optional[sa.Node]] = []
        self.pages_perms: List[List[int]] = [] # Each is Maps leaf_id -> image_idx
        self.pages_locks: List[Dict[int, int]] = {} # Each is leaf_id -> image_idx
        self.target_leaf_count: Optional[int] = None
        self.all_prefs: List[float] = []
        self.current_page_idx = 0
        
        self.page_W = 1000  # Internal logic size
        self.page_H = 1414  # ~A4 Aspect
        
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Left: Images
        left_layout = QVBoxLayout()
        self.load_btn = QPushButton("Load Folder")
        self.load_btn.clicked.connect(self.load_images_dialog)
        left_layout.addWidget(self.load_btn)
        
        select_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_images)
        self.select_none_btn = QPushButton("Select None")
        self.select_none_btn.clicked.connect(self.select_none_images)
        select_layout.addWidget(self.select_all_btn)
        select_layout.addWidget(self.select_none_btn)
        left_layout.addLayout(select_layout)
        
        self.image_list = QListWidget()
        self.image_list.setIconSize(QSize(100, 100))
        self.image_list.setDragEnabled(True)
        self.image_list.setDragDropMode(QListWidget.DragDropMode.DragOnly)
        left_layout.addWidget(self.image_list)
        
        # Center: Canvas
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setBackgroundBrush(QColor(50, 50, 50))
        
        # Right: Controls
        right_layout = QVBoxLayout()
        num_pages_row = QHBoxLayout()
        num_pages_label = QLabel("Number of Pages")
        self.num_pages_spin = QSpinBox()
        self.num_pages_spin.setMinimum(1)
        self.num_pages_spin.setValue(1)
        num_pages_row.addWidget(num_pages_label)
        num_pages_row.addWidget(self.num_pages_spin)
        right_layout.addLayout(num_pages_row)
        
        size_row = QHBoxLayout()
        size_label = QLabel("Slots per Page")
        self.slot_combo = QComboBox()
        self.slot_combo.setEnabled(False)
        self.slot_combo.currentIndexChanged.connect(self.on_slot_count_changed)
        size_row.addWidget(size_label)
        size_row.addWidget(self.slot_combo)
        right_layout.addLayout(size_row)

        self.optimize_btn = QPushButton("✨ Optimize Layout")
        self.optimize_btn.clicked.connect(self.start_optimization)
        self.optimize_btn.setEnabled(False)
        self.reset_btn = QPushButton("Reset Layout")
        self.reset_btn.clicked.connect(self.reset_layout)
        
        self.progress_bar = QProgressBar()
        
        right_layout.addWidget(self.optimize_btn)
        right_layout.addWidget(self.reset_btn)
        right_layout.addStretch()
        right_layout.addWidget(self.progress_bar)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(self.view)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(splitter)
        
        # Bottom: Navigation
        bottom_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous Page")
        self.prev_btn.clicked.connect(self.prev_page)
        self.page_label = QLabel("Page 1 / 1")
        self.next_btn = QPushButton("Next Page")
        self.next_btn.clicked.connect(self.next_page)
        bottom_layout.addWidget(self.prev_btn)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.page_label)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.next_btn)
        main_layout.addLayout(bottom_layout)

    def load_images_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.load_images(folder)

    def load_images(self, folder):
        self.image_paths = []
        self.all_prefs = []
        self.pages_roots = []
        self.pages_perms = []
        self.pages_locks = []
        self.target_leaf_count = None
        self.image_list.clear()
        
        folder_path = Path(folder)
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = sorted([p for p in folder_path.rglob("*") if p.suffix.lower() in exts])
        
        if not files:
            self.slot_combo.clear()
            self.slot_combo.setEnabled(False)
            self.target_leaf_count = None
            self.scene.clear()
            self.optimize_btn.setEnabled(False)
            return

        self.image_paths = files
        self.all_prefs = [sa.pref_aspect_for(p) for p in self.image_paths]
        self.update_slot_selector(len(self.image_paths))
        
        for idx, p in enumerate(self.image_paths):
            # Load thumbnail
            item = QListWidgetItem()
            # Efficient thumbnail loading?
            pix = QPixmap(str(p)).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            item.setIcon(QIcon(pix))
            item.setData(Qt.ItemDataRole.UserRole, idx) # Store index
            item.setText(p.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.image_list.addItem(item)
            
        self.init_trees()
        self.optimize_btn.setEnabled(bool(self.pages_roots))

    def update_slot_selector(self, total_images: int):
        if total_images <= 0:
            self.slot_combo.clear()
            self.slot_combo.setEnabled(False)
            self.target_leaf_count = None
            return

        options = []
        k = 1
        while k <= total_images:
            options.append(k)
            k *= 2

        preferred = self.target_leaf_count if self.target_leaf_count in options else options[-1]

        self.slot_combo.blockSignals(True)
        self.slot_combo.clear()
        for opt in options:
            self.slot_combo.addItem(str(opt), opt)
        self.slot_combo.setCurrentIndex(options.index(preferred))
        self.slot_combo.blockSignals(False)
        self.slot_combo.setEnabled(True)
        self.target_leaf_count = preferred

    def on_slot_count_changed(self, index: int):
        if index < 0:
            return
        value = self.slot_combo.itemData(index)
        if value is None:
            return
        count = int(value)
        if count == self.target_leaf_count:
            return
        self.target_leaf_count = count
        # Changing the slot count invalidates previous locks.
        self.init_trees()

    def init_trees(self):
        total_images = len(self.image_paths)
        if total_images == 0:
            self.optimize_btn.setEnabled(False)
            return

        if (
            self.target_leaf_count is None
            or not sa.is_power_of_two(self.target_leaf_count)
            or self.target_leaf_count > total_images
        ):
            self.target_leaf_count = sa.largest_power_of_two_leq(total_images)
            # Sync selector to the corrected value if it exists
            idx = self.slot_combo.findData(self.target_leaf_count)
            if idx >= 0:
                self.slot_combo.blockSignals(True)
                self.slot_combo.setCurrentIndex(idx)
                self.slot_combo.blockSignals(False)

        leaf_count = self.target_leaf_count
        num_pages = self.num_pages_spin.value()
        try:
            self.pages_roots = [sa.build_full_tree(leaf_count, seed=42 + i) for i in range(num_pages)]
        except AssertionError:
            self.pages_roots = []
            self.optimize_btn.setEnabled(False)
            return

        import random
        pool_indices = list(range(total_images))
        random.shuffle(pool_indices)
        self.pages_perms = [pool_indices[i*leaf_count:(i+1)*leaf_count] for i in range(num_pages)]
        self.pages_locks = [{} for _ in range(num_pages)]
        
        self.current_page_idx = 0
        self.update_page_nav()
        self.draw_layout()
        self.optimize_btn.setEnabled(bool(self.pages_roots))

    def draw_layout(self):
        self.scene.clear()
        if not self.pages_roots or self.current_page_idx >= len(self.pages_roots) or not self.pages_perms[self.current_page_idx]:
            return
        
        root = self.pages_roots[self.current_page_idx]
        perm = self.pages_perms[self.current_page_idx]
        locked = self.pages_locks[self.current_page_idx]
        
        margin = 20
        W, H = self.page_W, self.page_H
        in_W = W - 2*margin
        in_H = H - 2*margin
        
        boxes = sa.decode_region(root, margin, margin, in_W, in_H)
        
        gap = 10
        
        for leaf_id, (x, y, w, h) in boxes.items():
            # Apply gap
            rect_item = LeafItem(
                0, 0, w - gap, h - gap, 
                leaf_id, self
            )
            rect_item.setPos(x + gap/2, y + gap/2)
            
            if leaf_id >= len(perm):
                continue
            img_idx = perm[leaf_id]
            if img_idx < 0 or img_idx >= len(self.image_paths):
                continue
            
            # Load image to display in rect
            path = self.image_paths[img_idx]
            # We need to fit image
            pix = QPixmap(str(path))
            if not pix.isNull():
                 # Scale exactly to fit? Or AspectFit? 
                 # sa logic uses ImageOps.fit (crop to fill).
                 # Let's emulate crop to fill for preview.
                 scaled = pix.scaled(int(w-gap), int(h-gap), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
                 # Crop center
                 copy = scaled.copy(
                     (scaled.width() - int(w-gap)) // 2,
                     (scaled.height() - int(h-gap)) // 2,
                     int(w-gap), int(h-gap)
                 )
                 rect_item.pixmap_item.setPixmap(copy)
            
            if leaf_id in locked:
                rect_item.is_locked = True
                
            self.scene.addItem(rect_item)
            
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def handle_drop(self, leaf_id, img_idx):
        # User dropped image `img_idx` onto `leaf_id`.
        # Constraint: Image `img_idx` MUST be at `leaf_id`.
        if not self.pages_roots or self.current_page_idx >= len(self.pages_roots) or not self.pages_perms[self.current_page_idx]:
            return
        perm = self.pages_perms[self.current_page_idx]
        locked = self.pages_locks[self.current_page_idx]
        if leaf_id < 0 or leaf_id >= len(perm):
            return
        if img_idx < 0 or img_idx >= len(self.image_paths):
            return

        # Remove any previous lock that tied this image to a different leaf.
        for locked_leaf, locked_img in list(locked.items()):
            if locked_leaf != leaf_id and locked_img == img_idx:
                del locked[locked_leaf]

        locked[leaf_id] = img_idx

        if img_idx in perm:
            current_leaf_of_img = perm.index(img_idx)
            if current_leaf_of_img != leaf_id:
                perm[leaf_id], perm[current_leaf_of_img] = (
                    perm[current_leaf_of_img],
                    perm[leaf_id],
                )
        else:
            perm[leaf_id] = img_idx

        self.draw_layout()

    def start_optimization(self):
        if not self.pages_roots or not self.pages_perms:
            return

        num_pages = self.num_pages_spin.value()
        slots_per_page = self.target_leaf_count
        total_needed = num_pages * slots_per_page

        active_indices = []
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                idx = item.data(Qt.ItemDataRole.UserRole)
                active_indices.append(idx)

        if len(active_indices) < total_needed:
            QMessageBox.warning(self, "Not Enough Images", f"You need at least {total_needed} images for {num_pages} pages with {slots_per_page} slots each, but only {len(active_indices)} are selected.")
            return

        # Slice into chunks
        chunks = [active_indices[i*slots_per_page:(i+1)*slots_per_page] for i in range(num_pages)]
        swap_pool = active_indices[total_needed:]

        # Prepare for thread
        self.worker = OptimizationThread(
            chunks, swap_pool, self.pages_locks, self.all_prefs, self.image_paths, self.page_W, self.page_H
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished_optim.connect(self.finished_optimization)
        
        self.optimize_btn.setEnabled(False)
        self.worker.start()

    def finished_optimization(self, results):
        self.pages_perms = results
        self.update_page_nav()
        self.draw_layout()
        self.optimize_btn.setEnabled(True)
        QMessageBox.information(self, "Done", "Optimization Complete!")

    def reset_layout(self):
        for locks in self.pages_locks:
            locks.clear()
        self.init_trees() # Re-randomize

    def prev_page(self):
        if self.current_page_idx > 0:
            self.current_page_idx -= 1
            self.update_page_nav()
            self.draw_layout()

    def next_page(self):
        if self.current_page_idx < len(self.pages_roots) - 1:
            self.current_page_idx += 1
            self.update_page_nav()
            self.draw_layout()

    def update_page_nav(self):
        num_pages = len(self.pages_roots)
        if num_pages == 0:
            self.page_label.setText("Page 0 / 0")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
        else:
            self.page_label.setText(f"Page {self.current_page_idx + 1} / {num_pages}")
            self.prev_btn.setEnabled(self.current_page_idx > 0)
            self.next_btn.setEnabled(self.current_page_idx < num_pages - 1)

    def select_all_images(self):
        for i in range(self.image_list.count()):
            self.image_list.item(i).setCheckState(Qt.CheckState.Checked)

    def select_none_images(self):
        for i in range(self.image_list.count()):
            self.image_list.item(i).setCheckState(Qt.CheckState.Unchecked)

# Custom Drag for ListWidget
# We need to subclass QListWidget to support custom mime data easily, 
# or just rely on internal model?
# Actually, standard QListWidget drag puts mime data.
# Often easier to override startDrag in QListWidget
# But let's try to patch QListWidget instance methods or subclass properly.

# Monkey-patching for simplicity in one file
def startDrag(self, actions):
    drag = QDrag(self)
    items = self.selectedItems()
    if not items: return
    item = items[0]
    idx = item.data(Qt.ItemDataRole.UserRole)
    
    mime = QMimeData()
    mime.setData("application/x-image-idx", str(idx).encode())
    drag.setMimeData(mime)
    
    drag.exec(Qt.DropAction.CopyAction)

QListWidget.startDrag = startDrag


def main():
    app = QApplication(sys.argv)
    window = AlbumWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
