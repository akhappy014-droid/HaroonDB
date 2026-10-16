"use strict";

const form = document.getElementById("chatForm");
const input = document.getElementById("messageInput");
const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");
const sendBtn = document.getElementById("sendBtn");
const statusEl = document.getElementById("modelStatus");
const sidebar = document.getElementById("sidebar");
const overlay = document.getElementById("overlay");
const chatArea = document.getElementById("chatArea");

let historyItems = [];
let isSending = false;
let currentRequestId = 0;

function addMessage(role, content, extraClass = "") {
if (welcome) welcome.style.display = "none";

const row = document.createElement("div");
row.className = `message ${role} ${extraClass}`.trim();

const avatar = document.createElement("div");
avatar.className = "avatar";
avatar.textContent = role === "user" ? "أنت" : "H";

const body = document.createElement("div");
body.className = "message-content";
body.textContent = String(content ?? "");

row.append(avatar, body);
messages.appendChild(row);

scrollToBottom();

return row;

}

function scrollToBottom() {
if (!chatArea) return;

requestAnimationFrame(() => {
    chatArea.scrollTop = chatArea.scrollHeight;
});

}

function setSending(sending) {
isSending = sending;

if (sendBtn) {
    sendBtn.disabled = sending;
    sendBtn.textContent = sending ? "…" : "↑";
}

if (input) {
    input.disabled = sending;
}

}

function closeSidebar() {
if (sidebar) sidebar.classList.remove("open");
if (overlay) overlay.classList.remove("show");
}

async function readJSON(response) {
const rawText = await response.text();

if (!rawText.trim()) {
    throw new Error(
        `الخادم أعاد استجابة فارغة (HTTP ${response.status}). قد يكون النموذج بطيئًا أو توقف أثناء التوليد.`
    );
}

let data;

try {
    data = JSON.parse(rawText);
} catch {
    throw new Error(
        `استجابة الخادم ليست JSON صالحًا (HTTP ${response.status}). تحقق من سجلات Render.`
    );
}

if (!response.ok) {
    throw new Error(
        data.error ||
        data.message ||
        `حدث خطأ في الخادم (HTTP ${response.status}).`
    );
}

return data;

}

async function sendMessage(text) {
text = String(text || "").trim();

if (!text || isSending) return;

const requestId = ++currentRequestId;

addMessage("user", text);

// نحتفظ برسالة المستخدم ضمن السجل، ثم نرسل السياق السابق فقط.
const previousHistory = historyItems.slice(-8);

historyItems.push({
    role: "user",
    content: text
});

// منع تراكم سجل المحادثة بلا حدود.
if (historyItems.length > 20) {
    historyItems = historyItems.slice(-20);
}

input.value = "";
input.style.height = "auto";
setSending(true);

const typing = addMessage(
    "assistant",
    "يفكر Haroon...",
    "typing"
);

try {
    const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        body: JSON.stringify({
            message: text,
            history: previousHistory
        })
    });

    const data = await readJSON(response);

    if (requestId !== currentRequestId) return;

    if (typeof data.answer !== "string" || !data.answer.trim()) {
        throw new Error("لم يرجع النموذج إجابة نصية. تحقق من النموذج في الخادم.");
    }

    typing.remove();

    addMessage("assistant", data.answer.trim());

    historyItems.push({
        role: "assistant",
        content: data.answer.trim()
    });

    if (historyItems.length > 20) {
        historyItems = historyItems.slice(-20);
    }

} catch (error) {
    if (requestId !== currentRequestId) return;

    typing.remove();

    console.error("Haroon chat error:", error);

    addMessage(
        "assistant",
        `تعذر الحصول على الإجابة: ${error.message}`
    );

    // إزالة رسالة المستخدم الفاشلة من السياق كي لا تُرسل لاحقًا
    // وكأن النموذج أجاب عنها.
    if (
        historyItems.length &&
        historyItems[historyItems.length - 1].role === "user" &&
        historyItems[historyItems.length - 1].content === text
    ) {
        historyItems.pop();
    }

} finally {
    if (requestId === currentRequestId) {
        setSending(false);
        input.focus();
        scrollToBottom();
    }
}

}

// إرسال الرسالة من النموذج الأصلي.
if (form && input) {
form.addEventListener("submit", event => {
event.preventDefault();
sendMessage(input.value);
});

// Enter للإرسال، وShift + Enter لإضافة سطر جديد.
input.addEventListener("keydown", event => {
    if (
        event.key === "Enter" &&
        !event.shiftKey &&
        !event.isComposing
    ) {
        event.preventDefault();

        if (!isSending) {
            sendMessage(input.value);
        }
    }
});

input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height =
        Math.min(input.scrollHeight, 180) + "px";
});

}

// أزرار الأسئلة المقترحة الموجودة في index.html.
document.querySelectorAll("[data-prompt]").forEach(button => {
button.addEventListener("click", () => {
if (isSending) return;

    input.value = button.dataset.prompt || "";
    sendMessage(input.value);
});

});

// بدء محادثة جديدة.
const newChatBtn = document.getElementById("newChat");

if (newChatBtn) {
newChatBtn.addEventListener("click", () => {
// إبطال الطلب السابق في الواجهة.
currentRequestId++;

    messages.innerHTML = "";
    historyItems = [];

    if (welcome) welcome.style.display = "block";

    setSending(false);
    closeSidebar();

    input.value = "";
    input.style.height = "auto";
    input.focus();
});

}

// فتح القائمة الجانبية على الهاتف.
const menuBtn = document.getElementById("menuBtn");

if (menuBtn) {
menuBtn.addEventListener("click", () => {
if (sidebar) sidebar.classList.add("open");
if (overlay) overlay.classList.add("show");
});
}

if (overlay) {
overlay.addEventListener("click", closeSidebar);
}

// فحص حالة النموذج.
async function checkModelStatus() {
if (!statusEl) return;

try {
    const response = await fetch("/api/status", {
        method: "GET",
        headers: {
            "Accept": "application/json"
        },
        cache: "no-store"
    });

    const data = await readJSON(response);

    statusEl.textContent = data.ready
        ? "النموذج المدرّب جاهز"
        : "الواجهة جاهزة — الأوزان تحتاج تدريبًا";

    const dot = statusEl.previousElementSibling;

    if (dot) {
        dot.style.background = data.ready
            ? "#5cc79b"
            : "#e0a64b";
    }

} catch (error) {
    console.error("Haroon status error:", error);

    statusEl.textContent = "تعذر فحص حالة النموذج";

    const dot = statusEl.previousElementSibling;

    if (dot) {
        dot.style.background = "#e05252";
    }
}

}

checkModelStatus();
