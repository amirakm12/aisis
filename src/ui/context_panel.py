from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QTextEdit, QPushButton, QLineEdit,
    QListWidget, QListWidgetItem, QFrame, QComboBox, QScrollArea, QMenu, QAction, QFileDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QDateTime, QSize
from PyQt6.QtGui import QColor, QPalette, QIcon, QPixmap, QFont, QShortcut, QKeySequence
from .context_manager import ContextManager
import os
import time

class ContextPanel(QFrame):
    """
    Advanced Context Panel: modern, interactive, visually rich, and highly usable.
    """
    context_updated = pyqtSignal(dict)
    progress_updated = pyqtSignal(int, str)
    state_changed = pyqtSignal(str)
    notification = pyqtSignal(str, str)  # (message, level)

    def __init__(self, context_manager: ContextManager, parent=None):
        super().__init__(parent)
        self.context_manager = context_manager
        self.collapsed = False
        self.pinned_context = set()
        self.recent_actions = []
        self.notifications = []
        self.active_agent = None
        self.workspace_name = "Default Workspace"
        self._init_ui()
        self._init_shortcuts()
        self._restore_panel_state()

    def _init_ui(self):
        self.setObjectName("ContextPanel")
        self.setMinimumWidth(340)
        self.setMaximumWidth(520)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setStyleSheet("""
            QFrame#ContextPanel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #232946, stop:1 #121826);
                border-radius: 16px;
                border: 2px solid #6366f1;
                box-shadow: 0 4px 24px rgba(0,0,0,0.18);
            }
        """)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(16, 16, 16, 16)
        self.setLayout(self.layout)

        # Header: user/session info
        header = QHBoxLayout()
        self.avatar = QLabel()
        self.avatar.setPixmap(QPixmap(os.path.join(os.path.dirname(__file__), "user.png")).scaled(40, 40))
        self.avatar.setFixedSize(40, 40)
        header.addWidget(self.avatar)
        self.username = QLabel("User: Guest")
        self.username.setStyleSheet("font-weight: bold; font-size: 15px;")
        header.addWidget(self.username)
        self.session_time = QLabel(QDateTime.currentDateTime().toString("hh:mm AP"))
        self.session_time.setStyleSheet("color: #94a3b8; font-size: 12px;")
        header.addWidget(self.session_time)
        header.addStretch()
        self.collapse_btn = QPushButton(QIcon.fromTheme("go-up"), "")
        self.collapse_btn.setFixedWidth(28)
        self.collapse_btn.setToolTip("Collapse/Expand panel (Ctrl+K)")
        self.collapse_btn.clicked.connect(self.toggle_collapse)
        header.addWidget(self.collapse_btn)
        self.layout.addLayout(header)

        # Workspace/project name and switcher
        ws_layout = QHBoxLayout()
        self.workspace_label = QLabel(f"Workspace: {self.workspace_name}")
        self.workspace_label.setStyleSheet("font-size: 13px; color: #8b5cf6;")
        ws_layout.addWidget(self.workspace_label)
        self.workspace_switch = QComboBox()
        self.workspace_switch.addItems(["Default Workspace", "Project Alpha", "Project Beta"])
        self.workspace_switch.setToolTip("Switch workspace/project")
        self.workspace_switch.currentTextChanged.connect(self._on_workspace_switch)
        ws_layout.addWidget(self.workspace_switch)
        self.layout.addLayout(ws_layout)

        # Active agent/task display
        agent_layout = QHBoxLayout()
        self.agent_icon = QLabel()
        self.agent_icon.setPixmap(QPixmap(os.path.join(os.path.dirname(__file__), "agent.png")).scaled(24, 24))
        self.agent_icon.setFixedSize(24, 24)
        agent_layout.addWidget(self.agent_icon)
        self.agent_label = QLabel("Active Agent: None")
        self.agent_label.setStyleSheet("font-size: 13px; color: #06b6d4;")
        agent_layout.addWidget(self.agent_label)
        self.agent_status = QLabel("Idle")
        self.agent_status.setStyleSheet("color: #22c55e; font-weight: bold;")
        agent_layout.addWidget(self.agent_status)
        agent_layout.addStretch()
        self.layout.addLayout(agent_layout)

        # Multi-step progress bar
        self.progress_steps = ["Queued", "Running", "Postprocessing", "Complete"]
        self.progress_step_labels = []
        steps_layout = QHBoxLayout()
        for step in self.progress_steps:
            lbl = QLabel(step)
            lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
            self.progress_step_labels.append(lbl)
            steps_layout.addWidget(lbl)
        self.layout.addLayout(steps_layout)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.progress_steps)-1)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setStyleSheet("QProgressBar {background: #232946; border-radius: 6px;} QProgressBar::chunk {background: #6366f1; border-radius: 6px;}")
        self.layout.addWidget(self.progress_bar)

        # Search/filter for context/logs
        search_layout = QHBoxLayout()
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search context or logs...")
        self.search_box.textChanged.connect(self._on_search)
        search_layout.addWidget(self.search_box)
        self.pin_btn = QPushButton(QIcon.fromTheme("star"), "")
        self.pin_btn.setToolTip("Pin selected context")
        self.pin_btn.setFixedWidth(28)
        self.pin_btn.clicked.connect(self._pin_selected)
        search_layout.addWidget(self.pin_btn)
        self.layout.addLayout(search_layout)

        # Context list (virtualized)
        self.context_list = QListWidget()
        self.context_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.context_list.setToolTip("Session context (pinned items at top)")
        self.layout.addWidget(self.context_list)

        # Recent actions/commands
        self.actions_label = QLabel("Recent Actions:")
        self.actions_label.setStyleSheet("font-size: 13px; color: #f59e0b;")
        self.layout.addWidget(self.actions_label)
        self.actions_list = QListWidget()
        self.actions_list.setMaximumHeight(80)
        self.actions_list.setToolTip("Recent actions and commands")
        self.layout.addWidget(self.actions_list)

        # Live log area (virtualized)
        self.log_label = QLabel("Agent/Task Log:")
        self.log_label.setStyleSheet("font-size: 13px; color: #06b6d4;")
        self.layout.addWidget(self.log_label)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(120)
        self.log_area.setToolTip("Live log output. Auto-scroll enabled.")
        self.layout.addWidget(self.log_area)

        # Notifications/alerts area
        self.notif_label = QLabel()
        self.notif_label.setStyleSheet("font-size: 13px; color: #ef4444;")
        self.layout.addWidget(self.notif_label)

        # Quick actions
        quick_layout = QHBoxLayout()
        self.copy_btn = QPushButton(QIcon.fromTheme("edit-copy"), "Copy Context")
        self.copy_btn.setToolTip("Copy all context to clipboard")
        self.copy_btn.clicked.connect(self.copy_context)
        quick_layout.addWidget(self.copy_btn)
        self.clear_btn = QPushButton(QIcon.fromTheme("edit-clear"), "Clear Context")
        self.clear_btn.setToolTip("Clear all context and logs")
        self.clear_btn.clicked.connect(self.clear_context)
        quick_layout.addWidget(self.clear_btn)
        self.export_btn = QPushButton(QIcon.fromTheme("document-save"), "Export Context")
        self.export_btn.setToolTip("Export context and logs to file")
        self.export_btn.clicked.connect(self.export_context)
        quick_layout.addWidget(self.export_btn)
        self.feedback_btn = QPushButton(QIcon.fromTheme("help-about"), "Feedback")
        self.feedback_btn.setToolTip("Send feedback or report an issue")
        self.feedback_btn.clicked.connect(self.send_feedback)
        quick_layout.addWidget(self.feedback_btn)
        self.settings_btn = QPushButton(QIcon.fromTheme("preferences-system"), "Settings")
        self.settings_btn.setToolTip("Open settings dialog")
        quick_layout.addWidget(self.settings_btn)
        self.layout.addLayout(quick_layout)

        # High-contrast mode toggle
        self.high_contrast = False
        self.hc_btn = QPushButton("HC")
        self.hc_btn.setToolTip("Toggle high-contrast mode (Ctrl+H)")
        self.hc_btn.clicked.connect(self.toggle_high_contrast)
        self.layout.addWidget(self.hc_btn)

        self.setAccessibleName("Context Panel")
        self.setToolTip("Workspace context, progress, and logs")

        self.update_context_display()
        self._update_recent_actions()
        self._update_active_agent()
        self._update_notifications()

    def _init_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+K"), self, self.toggle_collapse)
        QShortcut(QKeySequence("Ctrl+F"), self, self.search_box.setFocus)
        QShortcut(QKeySequence("Ctrl+E"), self, self.export_context)
        QShortcut(QKeySequence("Ctrl+C"), self, self.copy_context)
        QShortcut(QKeySequence("Ctrl+L"), self, self.clear_context)

    def _restore_panel_state(self):
        # TODO: Load previous panel state (collapsed, scroll, etc.) from settings
        pass

    def _on_workspace_switch(self, name):
        self.workspace_name = name
        self.workspace_label.setText(f"Workspace: {name}")

    def _on_search(self, text):
        # Filter context and logs
        for i in range(self.context_list.count()):
            item = self.context_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())
        # Optionally filter logs
        # ...

    def _pin_selected(self):
        for item in self.context_list.selectedItems():
            self.pinned_context.add(item.text())
        self.update_context_display()

    def update_context_display(self):
        t0 = time.perf_counter()
        context = self.context_manager.session_context
        self.context_list.clear()
        # Pinned items at top
        pinned = [k for k in context if k in self.pinned_context]
        unpinned = [k for k in context if k not in self.pinned_context]
        for k in pinned + unpinned:
            item = QListWidgetItem(f"{k}: {context[k]}")
            if k in self.pinned_context:
                item.setBackground(QColor("#6366f1"))
                item.setForeground(QColor("#fff"))
            self.context_list.addItem(item)
        self.context_updated.emit(context)
        t1 = time.perf_counter()
        print(f"[PERF] Context update took {t1-t0:.4f}s")

    def set_progress(self, value: int, text: str = None, step: int = None):
        if step is not None and 0 <= step < len(self.progress_steps):
            self.progress_bar.setValue(step)
            for i, lbl in enumerate(self.progress_step_labels):
                lbl.setStyleSheet("color: #6366f1; font-weight: bold;" if i == step else "color: #94a3b8;")
        else:
            self.progress_bar.setValue(value)
        if text:
            self.progress_step_labels[-1].setText(text)
        self.progress_updated.emit(value, text or "")

    def set_state(self, state: str, agent: str = None):
        self.agent_status.setText(state)
        color = "#22c55e" if state.lower() == "active" else ("#ef4444" if state.lower() == "error" else "#94a3b8")
        self.agent_status.setStyleSheet(f"color: {color}; font-weight: bold;")
        if agent:
            self.agent_label.setText(f"Active Agent: {agent}")
        self.state_changed.emit(state)

    def update_context(self, event):
        self.context_manager.update_context(event)
        self.update_context_display()
        self._update_recent_actions(event)

    def log(self, message: str):
        t0 = time.perf_counter()
        self.log_area.append(message)
        self.log_area.moveCursor(self.log_area.textCursor().End)
        t1 = time.perf_counter()
        print(f"[PERF] Log update took {t1-t0:.4f}s")

    def clear_context(self):
        self.context_manager.session_context.clear()
        self.context_list.clear()
        self.log_area.clear()
        self.set_progress(0, "")
        self.set_state("Idle")
        self.pinned_context.clear()
        self._update_recent_actions()

    def export_context(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Export Context", "context.txt", "Text Files (*.txt)")
        if fname:
            with open(fname, "w") as f:
                f.write(f"[Workspace] {self.workspace_name}\n")
                for k, v in self.context_manager.session_context.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n[Log]\n")
                f.write(self.log_area.toPlainText())

    def copy_context(self):
        context = self.context_manager.session_context
        text = "\n".join(f"{k}: {v}" for k, v in context.items())
        self.context_display.setText(text)
        self.context_display.selectAll()
        self.context_display.copy()

    def send_feedback(self):
        # TODO: Open feedback dialog or send feedback
        self.notification.emit("Feedback dialog not implemented yet.", "info")

    def toggle_collapse(self):
        self.collapsed = not self.collapsed
        self.setVisible(not self.collapsed)
        # Optionally animate

    def _update_recent_actions(self, event=None):
        if event:
            action = event.get("action") or event.get("event")
            if action:
                timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
                self.recent_actions.append(f"{timestamp} - {action}")
                if len(self.recent_actions) > 10:
                    self.recent_actions = self.recent_actions[-10:]
        self.actions_list.clear()
        for a in reversed(self.recent_actions):
            self.actions_list.addItem(a)

    def _update_active_agent(self):
        # TODO: Set agent icon and label based on current agent
        pass

    def _update_notifications(self):
        if self.notifications:
            msg, level = self.notifications[-1]
            color = "#ef4444" if level == "error" else ("#f59e0b" if level == "warning" else "#22c55e")
            self.notif_label.setText(f"<span style='color:{color}'>{msg}</span>")
        else:
            self.notif_label.clear()

    def toggle_high_contrast(self):
        self.high_contrast = not self.high_contrast
        if self.high_contrast:
            self.setStyleSheet("QFrame#ContextPanel { background: #000; color: #fff; border: 2px solid #fff; }")
        else:
            self.setStyleSheet("QFrame#ContextPanel { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #232946, stop:1 #121826); border-radius: 16px; border: 2px solid #6366f1; box-shadow: 0 4px 24px rgba(0,0,0,0.18); }")

    def keyPressEvent(self, event):
        # Keyboard navigation for accessibility
        if event.key() == Qt.Key_Tab:
            self.focusNextChild()
        elif event.key() == Qt.Key_Backtab:
            self.focusPreviousChild()
        elif event.key() == Qt.Key_H and event.modifiers() & Qt.ControlModifier:
            self.toggle_high_contrast()
        else:
            super().keyPressEvent(event) 