let conversationHistory = [];

function appendMessage(role, content) {
    const chatBox = document.getElementById("chat-box");
    const emptyHint = chatBox.querySelector(".empty-hint");
    if (emptyHint) emptyHint.remove();

    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${role}`;
    msgDiv.innerHTML = `<div class="bubble">${escapeHtml(content)}</div>`;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

async function sendMessage() {
    const promptEl = document.getElementById("prompt");
    const sendBtn = document.getElementById("send-btn");
    const statusEl = document.getElementById("status");

    const content = promptEl.value.trim();
    if (!content) return;

    sendBtn.disabled = true;
    statusEl.textContent = "推理中...";

    conversationHistory.push({ role: "user", content: content });
    appendMessage("user", content);
    promptEl.value = "";

    try {
        const res = await fetch("/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ messages: conversationHistory }),
        });

        const data = await res.json();
        if (res.ok) {
            conversationHistory.push({ role: "assistant", content: data.generated });
            appendMessage("assistant", data.generated);
            statusEl.textContent = "";
        } else {
            appendMessage("assistant", "错误: " + (data.error || "未知错误"));
            statusEl.textContent = "失败";
        }
    } catch (e) {
        appendMessage("assistant", "请求失败: " + e.message);
        statusEl.textContent = "失败";
    } finally {
        sendBtn.disabled = false;
    }
}

function clearChat() {
    conversationHistory = [];
    const chatBox = document.getElementById("chat-box");
    chatBox.innerHTML = '<div class="empty-hint">开始对话吧</div>';
    document.getElementById("status").textContent = "";
}

document.getElementById("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});