import sys
import os
import random
from pathlib import Path
from typing import List, Dict, Optional

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, 
                             QGraphicsView, QGraphicsScene, QGraphicsRectItem, 
                             QGraphicsPixmapItem, QGraphicsTextItem, QFileDialog, QLabel, QProgressBar,
                             QSplitter, QMessageBox, QFrame, QComboBox, QSpinBox, QLineEdit, QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QMimeData, QPointF
from PyQt6.QtGui import QPixmap, QDrag, QImage, QPainter, QColor, QPen, QIcon, QTextDocument

import sa_advanced as sa
from PIL import Image, ImageOps

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ImageListEntry(QWidget):
    def __init__(self, pixmap, filename, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        
        self.thumb = QLabel()
        self.thumb.setPixmap(pixmap)
        self.thumb.setFixedSize(60, 60)
        self.thumb.setScaledContents(True)
        # Allow clicks to pass through to the list item for selection/dragging
        self.thumb.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(self.thumb)
        
        self.name_label = QLabel(filename)
        self.name_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(self.name_label, 1)
        
        self.pool_cb = QCheckBox("Pool")
        self.pool_cb.setChecked(True)
        layout.addWidget(self.pool_cb)
        
        self.mandatory_cb = QCheckBox("Mandatory")
        layout.addWidget(self.mandatory_cb)
        
        self.pool_cb.toggled.connect(self.on_pool_toggled)
        
    def on_pool_toggled(self, checked):
        self.mandatory_cb.setEnabled(checked)
        if not checked:
            self.mandatory_cb.setChecked(False)

class OptimizationThread(QThread):
    progress = pyqtSignal(int, int)
    finished_optim = pyqtSignal(list, list)
    
    def __init__(self, chunks, swap_pool, pages_locks, all_prefs, image_paths, page_W, page_H, title, pages_roots, steps, mode, crop_to_fit=True):
        super().__init__()
        self.chunks = chunks
        self.swap_pool = swap_pool.copy()
        self.pages_locks = pages_locks
        self.all_prefs = all_prefs
        self.image_paths = image_paths
        self.page_W = page_W
        self.page_H = page_H
        self.title = title
        self.pages_roots = pages_roots
        self.steps = steps
        self.mode = mode
        self.crop_to_fit = crop_to_fit
        
    def run(self):
        if self.mode == "es":
            results, final_pool, energy_history = sa.optimize_es(
                roots=self.pages_roots,
                page_W=self.page_W,
                page_H=self.page_H,
                all_images=self.image_paths,
                all_prefs=self.all_prefs,
                initial_perms=self.chunks,
                swap_pool=self.swap_pool,
                locked_leaves=self.pages_locks,
                steps=self.steps,
                progress_callback=self.emit_progress,
                title=self.title
            )
        elif self.mode == "linear":
            new_roots, results, final_pool, energy_history = sa.optimize_linear_partition(
                roots=self.pages_roots,
                page_W=self.page_W,
                page_H=self.page_H,
                all_images=self.image_paths,
                all_prefs=self.all_prefs,
                initial_perms=self.chunks,
                swap_pool=self.swap_pool,
                locked_leaves=self.pages_locks,
                steps=self.steps,
                progress_callback=self.emit_progress,
                title=self.title
            )
            # Update roots in place
            for i in range(len(self.pages_roots)):
                self.pages_roots[i] = new_roots[i]
        else:
            results, final_pool, energy_history = sa.anneal_global(
                roots=self.pages_roots,
                page_W=self.page_W,
                page_H=self.page_H,
                all_images=self.image_paths,
                all_prefs=self.all_prefs,
                initial_perms=self.chunks,
                swap_pool=self.swap_pool,
                locked_leaves=self.pages_locks,
                steps=self.steps,
                progress_callback=self.emit_progress,
                title=self.title
            )
        self.finished_optim.emit(results, energy_history)
        
    def emit_progress(self, step, total):
        self.progress.emit(step, total)

class ExportThread(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, output_path, pages_roots, pages_perms, image_paths, title, crop_to_fit=True):
        super().__init__()
        self.output_path = output_path
        self.pages_roots = pages_roots
        self.pages_perms = pages_perms
        self.image_paths = image_paths
        self.title = title
        self.crop_to_fit = crop_to_fit
        self.export_W = 2480
        self.export_H = 3508
        
    def run(self):
        try:
            pdf_pages = []
            total = len(self.pages_roots)
            for i, (root, perm) in enumerate(zip(self.pages_roots, self.pages_perms)):
                if not root or not perm:
                    continue
                
                page_img = sa.render_page(
                    root=root,
                    page_W=self.export_W,
                    page_H=self.export_H,
                    images=self.image_paths,
                    perm=perm,
                    page_margin_px=120,
                    gap_px=40,
                    title=self.title,
                    crop_to_fit=self.crop_to_fit
                )
                pdf_pages.append(page_img)
                self.progress.emit(i + 1, total)
            
            if pdf_pages:
                pdf_pages[0].save(
                    self.output_path, "PDF", resolution=300.0,
                    save_all=True, append_images=pdf_pages[1:]
                )
                self.finished.emit(True, f"Successfully saved to {self.output_path}")
            else:
                self.finished.emit(False, "No pages to export.")
        except Exception as e:
            self.finished.emit(False, str(e))

class LeafItem(QGraphicsRectItem):
    def __init__(self, x, y, w, h, page_idx, leaf_id, parent_gui):
        super().__init__(x, y, w, h)
        self.page_idx = page_idx
        self.leaf_id = leaf_id
        self.parent_gui = parent_gui
        self.setAcceptDrops(True)
        self.setPen(QPen(Qt.GlobalColor.black))
        self.setBrush(QColor(255, 255, 255))
        
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
            self.parent_gui.handle_drop(self.page_idx, self.leaf_id, img_idx)
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
        self.current_folder: Optional[Path] = None
        self.image_metadata: List[sa.ImageMetadata] = []
        self.pages_roots: List[Optional[sa.Node]] = []
        self.pages_perms: List[List[int]] = [] # Each is Maps leaf_id -> image_idx
        self.pages_locks: List[Dict[int, int]] = [] # Each is leaf_id -> image_idx
        self.target_leaf_count: Optional[int] = None
        self.all_prefs: List[float] = []
        self.current_page_idx = 0
        self.view_mode = "single" # "single" or "grid"
        
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
        self.num_pages_spin.valueChanged.connect(self.on_page_count_changed)
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
        
        title_row = QHBoxLayout()
        title_label = QLabel("Page Title")
        self.title_edit = QLineEdit()
        self.title_edit.setText("default title")
        title_row.addWidget(title_label)
        title_row.addWidget(self.title_edit)
        right_layout.addLayout(title_row)

        mode_row = QHBoxLayout()
        mode_label = QLabel("Optimization Algorithm")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Simulated Annealing", "sa")
        self.mode_combo.addItem("Evolutionary Strategy", "es")
        self.mode_combo.addItem("Linear Partitioning (Instant)", "linear")
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.mode_combo)
        right_layout.addLayout(mode_row)

        steps_row = QHBoxLayout()
        steps_label = QLabel("Annealing Steps")
        self.steps_spin = QSpinBox()
        self.steps_spin.setMinimum(1000)
        self.steps_spin.setMaximum(100000)
        self.steps_spin.setValue(5000)
        self.steps_spin.setSingleStep(1000)
        steps_row.addWidget(steps_label)
        steps_row.addWidget(self.steps_spin)
        right_layout.addLayout(steps_row)

        self.crop_checkbox = QCheckBox("Crop Images to Fill Slots")
        self.crop_checkbox.setChecked(True)
        self.crop_checkbox.stateChanged.connect(self.draw_layout)
        right_layout.addWidget(self.crop_checkbox)

        self.optimize_btn = QPushButton("✨ Optimize Layout")
        self.optimize_btn.clicked.connect(self.start_optimization)
        self.optimize_btn.setEnabled(False)
        self.reset_btn = QPushButton("Reset Layout")
        self.reset_btn.clicked.connect(self.reset_layout)
        self.export_btn = QPushButton("Save as PDF")
        self.export_btn.clicked.connect(self.export_pdf_dialog)
        self.export_btn.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        
        right_layout.addWidget(self.optimize_btn)
        right_layout.addWidget(self.reset_btn)
        right_layout.addWidget(self.export_btn)
        right_layout.addStretch()

        # Plot Placeholder
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        right_layout.addWidget(self.plot_container)
        self.figure = None
        self.canvas = None
        self.ax = None

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
        
        self.grid_view_btn = QPushButton("Grid View")
        self.grid_view_btn.setCheckable(True)
        self.grid_view_btn.clicked.connect(self.toggle_view_mode)
        
        bottom_layout.addWidget(self.prev_btn)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.page_label)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.grid_view_btn)
        bottom_layout.addWidget(self.next_btn)
        main_layout.addLayout(bottom_layout)

    def load_images_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.load_images(folder)

    def load_images(self, folder):
        self.image_paths = []
        self.current_folder = Path(folder)
        self.image_metadata = []
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
        
        # Task 2: Image Metadata Caching
        self.image_metadata = sa.batch_process_images(self.image_paths)
        self.all_prefs = [m.pref_aspect for m in self.image_metadata]
        
        self.update_slot_selector(len(self.image_paths))
        
        for idx, p in enumerate(self.image_paths):
            item = QListWidgetItem(self.image_list)
            item.setData(Qt.ItemDataRole.UserRole, idx)
            item.setSizeHint(QSize(200, 70))
            
            pix = QPixmap(str(p)).scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            widget = ImageListEntry(pix, p.name)
            self.image_list.setItemWidget(item, widget)
            
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

    def on_page_count_changed(self, value):
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
        max_possible_pages = max(1, total_images // leaf_count)
        requested_pages = self.num_pages_spin.value()
        if requested_pages > max_possible_pages:
            self.num_pages_spin.blockSignals(True)
            self.num_pages_spin.setValue(max_possible_pages)
            self.num_pages_spin.blockSignals(False)
            num_pages = max_possible_pages
        else:
            num_pages = requested_pages
        try:
            self.pages_roots = [sa.build_full_tree(leaf_count, seed=42 + i) for i in range(num_pages)]
        except AssertionError:
            self.pages_roots = []
            self.optimize_btn.setEnabled(False)
            return

        pool_indices = list(range(total_images))
        random.shuffle(pool_indices)
        self.pages_perms = [pool_indices[i*leaf_count:(i+1)*leaf_count] for i in range(num_pages)]
        self.pages_locks = [{} for _ in range(num_pages)]
        
        self.current_page_idx = 0
        self.update_page_nav()
        self.draw_layout()
        self.optimize_btn.setEnabled(bool(self.pages_roots))
        self.export_btn.setEnabled(bool(self.pages_roots))

    def draw_layout(self):
        self.scene.clear()
        if not self.pages_roots:
            return
        
        if self.view_mode == "single":
            if self.current_page_idx >= len(self.pages_roots):
                return
            self.render_page_to_scene(self.current_page_idx, 0, 0)
        else:
            # Grid View
            cols = 3
            visual_gap = 100
            for i in range(len(self.pages_roots)):
                row = i // cols
                col = i % cols
                x_offset = col * (self.page_W + visual_gap)
                y_offset = row * (self.page_H + visual_gap)
                self.render_page_to_scene(i, x_offset, y_offset)
                
                # Add page label
                label = QGraphicsTextItem(f"Page {i + 1}")
                font = label.font()
                font.setPixelSize(40)
                font.setBold(True)
                label.setFont(font)
                label.setDefaultTextColor(QColor(200, 200, 200))
                label.setPos(x_offset, y_offset - 60)
                self.scene.addItem(label)

        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def render_page_to_scene(self, page_idx, x_offset, y_offset):
        root = self.pages_roots[page_idx]
        perm = self.pages_perms[page_idx]
        locked = self.pages_locks[page_idx]
        
        margin = 20
        W, H = self.page_W, self.page_H
        title_height = int(H * 0.1)
        in_W = W - 2*margin
        in_H = H - 2*margin - title_height
        
        # Draw page background
        bg_rect = QGraphicsRectItem(x_offset, y_offset, W, H)
        bg_rect.setBrush(QColor(255, 255, 255))
        bg_rect.setPen(QPen(Qt.GlobalColor.black, 2))
        self.scene.addItem(bg_rect)

        boxes = sa.decode_region(root, margin, margin + title_height, in_W, in_H)
        
        # Draw title
        title_text = self.title_edit.text()
        text_item = QGraphicsTextItem()
        text_document = QTextDocument()
        text_document.setHtml(f"<div style='text-align: center;'>{title_text}</div>")
        text_item.setDocument(text_document)
        text_item.setPos(x_offset + margin, y_offset + margin)
        text_item.setTextWidth(W - 2*margin)
        font = text_item.font()
        font.setPixelSize(int(title_height * 0.4))
        text_item.setFont(font)
        text_item.setDefaultTextColor(QColor(0, 0, 0))
        self.scene.addItem(text_item)
        
        gap = 10
        
        for leaf_id, (x, y, w, h) in boxes.items():
            # Apply gap
            rect_item = LeafItem(
                0, 0, w - gap, h - gap, 
                page_idx, leaf_id, self
            )
            rect_item.setPos(x_offset + x + gap/2, y_offset + y + gap/2)
            
            if leaf_id >= len(perm):
                continue
            img_idx = perm[leaf_id]
            if img_idx < 0 or img_idx >= len(self.image_paths):
                continue
            
            # Load image to display in rect
            path = self.image_paths[img_idx]
            pix = QPixmap(str(path))
            if not pix.isNull():
                 if self.crop_checkbox.isChecked():
                    scaled = pix.scaled(int(w-gap), int(h-gap), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
                    copy = scaled.copy(
                        (scaled.width() - int(w-gap)) // 2,
                        (scaled.height() - int(h-gap)) // 2,
                        int(w-gap), int(h-gap)
                    )
                    rect_item.pixmap_item.setPixmap(copy)
                 else:
                    scaled = pix.scaled(int(w-gap), int(h-gap), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    rect_item.pixmap_item.setPixmap(scaled)
                    rect_item.pixmap_item.setPos((w - gap - scaled.width()) / 2, (h - gap - scaled.height()) / 2)
            
            if leaf_id in locked:
                rect_item.is_locked = True
                
            self.scene.addItem(rect_item)

    def handle_drop(self, page_idx, leaf_id, img_idx):
        # User dropped image `img_idx` onto `leaf_id` of `page_idx`.
        # Constraint: Image `img_idx` MUST be at `leaf_id`.
        if not self.pages_roots or page_idx >= len(self.pages_roots) or not self.pages_perms[page_idx]:
            return
        perm = self.pages_perms[page_idx]
        locked = self.pages_locks[page_idx]
        if leaf_id < 0 or leaf_id >= len(perm):
            return
        if img_idx < 0 or img_idx >= len(self.image_paths):
            return

        # Remove any previous lock that tied this image to a different leaf on the SAME page.
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

        # Relaunch: Regenerate trees to avoid local minima, preserving locks
        self.pages_roots = [sa.build_full_tree(slots_per_page, seed=random.randint(0, 1000000)) for _ in range(num_pages)]

        # Step A: Identify locked indices (implicitly mandatory)
        locked_indices = set()
        for p_locks in self.pages_locks:
            for img_idx in p_locks.values():
                locked_indices.add(img_idx)

        # Step B: Gather mandatory and optional images from the list
        mandatory_indices = set(locked_indices)
        optional_indices = []

        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            widget = self.image_list.itemWidget(item)
            if isinstance(widget, ImageListEntry):
                idx = item.data(Qt.ItemDataRole.UserRole)
                if widget.mandatory_cb.isChecked():
                    mandatory_indices.add(idx)
                elif widget.pool_cb.isChecked() and idx not in locked_indices:
                    optional_indices.append(idx)

        # Step C: Validation
        if len(mandatory_indices) > total_needed:
            QMessageBox.warning(self, "Too Many Mandatory Images", 
                                f"You have {len(mandatory_indices)} mandatory images (including locked ones) but only {total_needed} slots available.")
            return

        if len(mandatory_indices) + len(optional_indices) < total_needed:
            QMessageBox.warning(self, "Not Enough Images", 
                                f"You need {total_needed} images, but only {len(mandatory_indices) + len(optional_indices)} are available in the pool.")
            return

        # Step D: Selection
        # Start with mandatory images
        active_indices = list(mandatory_indices)
        # Fill the rest with optional images (sequential selection)
        slots_remaining = total_needed - len(active_indices)
        if slots_remaining > 0:
            active_indices.extend(optional_indices[:slots_remaining])

        # Initialize buckets
        chunks = [[] for _ in range(num_pages)]
        assigned = set()

        # Identify and assign locked images to their specific pages
        for p in range(num_pages):
            for leaf_id, img_idx in self.pages_locks[p].items():
                if img_idx in active_indices:
                    chunks[p].append(img_idx)
                    assigned.add(img_idx)

        # Distribute remaining active images into chunks
        free_active = [idx for idx in active_indices if idx not in assigned]
        for p in range(num_pages):
            while len(chunks[p]) < slots_per_page and free_active:
                chunks[p].append(free_active.pop(0))

        # Create swap pool from any remaining optional images not in active_indices
        swap_pool = optional_indices[slots_remaining:]

        # Prepare for thread
        title = self.title_edit.text()
        steps = self.steps_spin.value()
        mode = self.mode_combo.currentData()
        self.worker = OptimizationThread(
            chunks, swap_pool, self.pages_locks, self.all_prefs, self.image_paths, self.page_W, self.page_H, title, self.pages_roots, steps, mode,
            crop_to_fit=self.crop_checkbox.isChecked()
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished_optim.connect(self.on_optim_finished)
        
        self.optimize_btn.setEnabled(False)
        self.worker.start()

    def on_optim_finished(self, results, energy_history):
        self.pages_perms = results
        self.update_page_nav()
        self.draw_layout()
        self.optimize_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.update_energy_plot(energy_history)

    def update_energy_plot(self, history):
        if self.canvas is None:
            self.figure = Figure(figsize=(5, 3))
            self.canvas = FigureCanvas(self.figure)
            self.plot_layout.addWidget(self.canvas)
            self.ax = self.figure.add_subplot(111)
        else:
            self.ax.clear()

        self.ax.plot(history)
        self.ax.set_title("Energy Curve")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Energy")
        self.figure.tight_layout()
        self.canvas.draw()

    def export_pdf_dialog(self):
        if not self.current_folder:
            return
            
        pdf_dir = Path("pdfs")
        pdf_dir.mkdir(exist_ok=True)
        
        base_name = self.current_folder.name
        i = 1
        while True:
            path = pdf_dir / f"{base_name}_{i}.pdf"
            if not path.exists():
                break
            i += 1
            
        self.export_btn.setEnabled(False)
        self.optimize_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        title = self.title_edit.text()
        self.export_worker = ExportThread(
            str(path), self.pages_roots, self.pages_perms, self.image_paths, title,
            crop_to_fit=self.crop_checkbox.isChecked()
        )
        self.export_worker.progress.connect(self.progress_bar.setValue)
        self.export_worker.finished.connect(self.on_export_finished)
        self.export_worker.start()

    def on_export_finished(self, success, message):
        self.export_btn.setEnabled(True)
        self.optimize_btn.setEnabled(True)
        if not success:
            QMessageBox.critical(self, "Export Failed", message)

    def reset_layout(self):
        for locks in self.pages_locks:
            locks.clear()
        self.init_trees() # Re-randomize

    def toggle_view_mode(self):
        if self.grid_view_btn.isChecked():
            self.view_mode = "grid"
            self.grid_view_btn.setText("Single View")
        else:
            self.view_mode = "single"
            self.grid_view_btn.setText("Grid View")
        self.update_page_nav()
        self.draw_layout()

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
        elif self.view_mode == "grid":
            self.page_label.setText(f"All Pages ({num_pages})")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
        else:
            self.page_label.setText(f"Page {self.current_page_idx + 1} / {num_pages}")
            self.prev_btn.setEnabled(self.current_page_idx > 0)
            self.next_btn.setEnabled(self.current_page_idx < num_pages - 1)

    def select_all_images(self):
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            widget = self.image_list.itemWidget(item)
            if isinstance(widget, ImageListEntry):
                widget.pool_cb.setChecked(True)

    def select_none_images(self):
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            widget = self.image_list.itemWidget(item)
            if isinstance(widget, ImageListEntry):
                widget.pool_cb.setChecked(False)

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
