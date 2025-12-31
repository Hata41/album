import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, 
                             QGraphicsView, QGraphicsScene, QGraphicsRectItem, 
                             QGraphicsPixmapItem, QFileDialog, QLabel, QProgressBar,
                             QSplitter, QMessageBox, QFrame, QComboBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QMimeData, QPointF
from PyQt6.QtGui import QPixmap, QDrag, QImage, QPainter, QColor, QPen, QIcon

import sa_advanced as sa
from PIL import Image, ImageOps

class OptimizationThread(QThread):
    progress = pyqtSignal(int, int)
    finished_optim = pyqtSignal(list)
    
    def __init__(self, root, page_W, page_H, images, all_prefs, locked_leaves, subset_to_global):
        super().__init__()
        self.root = root
        self.page_W = page_W
        self.page_H = page_H
        self.images = images
        self.all_prefs = all_prefs
        self.locked_leaves = locked_leaves
        self.subset_to_global = subset_to_global
        
    def run(self):
        # We don't need snapshots for the optimization run itself unless we want to display them
        # Let's just run it.
        perm, _ = sa.anneal_with_snapshots(
            root=self.root,
            page_W=self.page_W,
            page_H=self.page_H,
            all_images=self.images,
            all_prefs=self.all_prefs,
            page_margin_px=50, # Scaled down for GUI? Or full size? 
                          # Ideally we use full size logic but simple coordinates.
            gap_px=20,
            steps=5000, # Faster for interactive
            snapshots_count=0,
            seed=42, # Make random?
            desc="Optimizing",
            locked_leaves=self.locked_leaves,
            progress_callback=self.emit_progress
        )
        global_perm = [self.subset_to_global[subset_idx] for subset_idx in perm]
        self.finished_optim.emit(global_perm)
        
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
        self.root: Optional[sa.Node] = None
        self.perm: List[int] = [] # Maps leaf_id -> image_idx
        self.locked_leaves: Dict[int, int] = {} # leaf_id -> image_idx
        self.target_leaf_count: Optional[int] = None
        self.all_prefs: List[float] = []
        
        self.page_W = 1000  # Internal logic size
        self.page_H = 1414  # ~A4 Aspect
        
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
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
        size_row = QHBoxLayout()
        size_label = QLabel("Target slots")
        self.slot_combo = QComboBox()
        self.slot_combo.setEnabled(False)
        self.slot_combo.currentIndexChanged.connect(self.on_slot_count_changed)
        size_row.addWidget(size_label)
        size_row.addWidget(self.slot_combo)

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
        
        layout.addWidget(splitter)

    def load_images_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.load_images(folder)

    def load_images(self, folder):
        self.image_paths = []
        self.all_prefs = []
        self.root = None
        self.perm = []
        self.locked_leaves = {}
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
            
        self.init_tree()
        self.optimize_btn.setEnabled(self.root is not None)

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
        self.locked_leaves = {}
        self.init_tree()

    def init_tree(self):
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
        try:
            self.root = sa.build_full_tree(leaf_count, seed=42)
        except AssertionError:
            self.root = None
            self.optimize_btn.setEnabled(False)
            return

        import random
        pool_indices = list(range(total_images))
        random.shuffle(pool_indices)
        self.perm = pool_indices[:leaf_count]
        self.locked_leaves = {}
        
        self.draw_layout()
        self.optimize_btn.setEnabled(self.root is not None)

    def draw_layout(self):
        self.scene.clear()
        if not self.root or not self.perm: return
        
        margin = 20
        W, H = self.page_W, self.page_H
        in_W = W - 2*margin
        in_H = H - 2*margin
        
        boxes = sa.decode_region(self.root, margin, margin, in_W, in_H)
        
        gap = 10
        
        for leaf_id, (x, y, w, h) in boxes.items():
            # Apply gap
            rect_item = LeafItem(
                0, 0, w - gap, h - gap, 
                leaf_id, self
            )
            rect_item.setPos(x + gap/2, y + gap/2)
            
            if leaf_id >= len(self.perm):
                continue
            img_idx = self.perm[leaf_id]
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
            
            if leaf_id in self.locked_leaves:
                rect_item.is_locked = True
                
            self.scene.addItem(rect_item)
            
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def handle_drop(self, leaf_id, img_idx):
        # User dropped image `img_idx` onto `leaf_id`.
        # Constraint: Image `img_idx` MUST be at `leaf_id`.
        if not self.root or not self.perm:
            return
        if leaf_id < 0 or leaf_id >= len(self.perm):
            return
        if img_idx < 0 or img_idx >= len(self.image_paths):
            return

        # Remove any previous lock that tied this image to a different leaf.
        for locked_leaf, locked_img in list(self.locked_leaves.items()):
            if locked_leaf != leaf_id and locked_img == img_idx:
                del self.locked_leaves[locked_leaf]

        self.locked_leaves[leaf_id] = img_idx

        if img_idx in self.perm:
            current_leaf_of_img = self.perm.index(img_idx)
            if current_leaf_of_img != leaf_id:
                self.perm[leaf_id], self.perm[current_leaf_of_img] = (
                    self.perm[current_leaf_of_img],
                    self.perm[leaf_id],
                )
        else:
            self.perm[leaf_id] = img_idx

        self.draw_layout()

    def start_optimization(self):
        if not self.root or not self.perm:
            return

        active_indices = []
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                idx = item.data(Qt.ItemDataRole.UserRole)
                active_indices.append(idx)

        if not active_indices:
            QMessageBox.warning(self, "No Images", "Please select at least one image.")
            return

        active_count = len(active_indices)
        if active_count < self.target_leaf_count:
            self.target_leaf_count = sa.largest_power_of_two_leq(active_count)
            self.init_tree()  # rebuild tree with new count

        active_paths = [self.image_paths[idx] for idx in active_indices]
        active_prefs = [self.all_prefs[idx] for idx in active_indices]
        subset_to_global = {i: idx for i, idx in enumerate(active_indices)}

        # Filter and remap locked_leaves
        new_locked = {}
        for leaf_id, img_idx in self.locked_leaves.items():
            if img_idx in active_indices:
                subset_idx = active_indices.index(img_idx)
                new_locked[leaf_id] = subset_idx

        self.worker = OptimizationThread(
            self.root, self.page_W, self.page_H,
            active_paths, active_prefs, new_locked, subset_to_global
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished_optim.connect(self.on_optim_finished)
        
        self.optimize_btn.setEnabled(False)
        self.worker.start()

    def on_optim_finished(self, perm):
        self.perm = perm
        self.draw_layout()
        self.optimize_btn.setEnabled(True)
        QMessageBox.information(self, "Done", "Optimization Complete!")

    def reset_layout(self):
        self.locked_leaves = {}
        self.init_tree() # Re-randomize

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
