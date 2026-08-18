async function sendPrompt() {
    const promptEl = document.getElementById("prompt");
    const sendBtn = document.getElementById("send-btn");
    const statusEl = document.getElementById("status");
    const outputEl = document.getElementById("output");

    const prompt = promptEl.value.trim();
    if (!prompt) return;

    sendBtn.disabled = true;
    statusEl.textContent = "推理中...";

    try {
        const res = await fetch("/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt }),
        });

        const data = await res.json();
        if (res.ok) {
            outputEl.textContent = data.generated;
            statusEl.textContent = "完成";
        } else {
            outputEl.textContent = "错误: " + (data.error || "未知错误");
            statusEl.textContent = "失败";
        }
    } catch (e) {
        outputEl.textContent = "请求失败: " + e.message;
        statusEl.textContent = "失败";
    } finally {
        sendBtn.disabled = false;
    }
}

document.getElementById("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendPrompt();
    }
});
