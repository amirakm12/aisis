from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QListWidget, QListWidgetItem
from PyQt6.QtCore import Qt, pyqtSignal
from src.agents.llm_client import LLMClient

class ChatPanel(QWidget):
    """
    Conversational memory and multi-turn dialog panel.
    Displays chat history, supports workflow refinement, undo, and branching.
    Integrates with LLMClient for AI responses and suggestions.
    """
    message_sent = pyqtSignal(str)
    workflow_refined = pyqtSignal(list)
    undo_requested = pyqtSignal()
    branch_requested = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.llm = LLMClient()
        self.history = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.chat_list = QListWidget()
        layout.addWidget(self.chat_list)
        input_layout = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type a message or command...")
        self.input_box.returnPressed.connect(self._on_send)
        input_layout.addWidget(self.input_box)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._on_send)
        input_layout.addWidget(self.send_btn)
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self._on_undo)
        input_layout.addWidget(self.undo_btn)
        self.branch_btn = QPushButton("Branch")
        self.branch_btn.clicked.connect(self._on_branch)
        input_layout.addWidget(self.branch_btn)
        layout.addLayout(input_layout)

    def add_message(self, text, sender="user"):
        item = QListWidgetItem(f"{sender.title()}: {text}")
        if sender == "ai":
            item.setForeground(Qt.GlobalColor.blue)
        self.chat_list.addItem(item)
        self.chat_list.scrollToBottom()
        self.history.append({"sender": sender, "text": text})

    def _on_send(self):
        text = self.input_box.text().strip()
        if not text:
            return
        self.add_message(text, sender="user")
        self.input_box.clear()
        # Get AI response/suggestion
        context = {}
        history = [m["text"] for m in self.history]
        ai_steps = self.llm.llm_parse(text, context, history)
        if ai_steps:
            self.add_message(f"Suggested workflow: {ai_steps}", sender="ai")
            self.workflow_refined.emit(ai_steps)
        else:
            self.add_message("Sorry, I couldn't parse that.", sender="ai")
        self.message_sent.emit(text)

    def _on_undo(self):
        self.undo_requested.emit()
        self.add_message("Undo requested.", sender="user")

    def _on_branch(self):
        # For demo: branch from last AI suggestion
        for m in reversed(self.history):
            if m["sender"] == "ai" and "workflow" in m["text"]:
                # Extract steps from message
                import re, json
                match = re.search(r"\[(.*)\]", m["text"])
                if match:
                    try:
                        steps = json.loads("[" + match.group(1) + "]")
                        self.branch_requested.emit(steps)
                        self.add_message("Branched workflow.", sender="user")
                        return
                    except Exception:
                        pass
        self.add_message("No workflow to branch from.", sender="ai") 