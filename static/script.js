// Script: Haroon-2.5-HK Chat
// Location: static/script.js

"use strict";

document.addEventListener("DOMContentLoaded", () => {
    const chatForm =
        document.getElementById("chatForm") ||
        document.getElementById("messageForm");

    const messageInput =
        document.getElementById("messageInput") ||
        document.getElementById("userInput");

    const chatMessages =
        document.getElementById("chatMessages") ||
        document.getElementById("messages");

    const sendButton =
        document.getElementById("sendButton") ||
        document.getElementById("sendBtn");

    const statusElement =
        document.getElementById("modelStatus");

    const conversation = [];

    let sending = false;

    // إنشاء عنصر رسالة بأمان
    function addMessage(role, content) {
        if (!chatMessages) {
            console.error("لم يتم العثور على عنصر عرض الرسائل.");
            return;
        }

        const message = document.createElement("div");
        message.className = `message ${role}-message`;

        const bubble = document.createElement("div");
        bubble.className = "message-content";
        bubble.textContent = content;

        message.appendChild(bubble);
        chatMessages.appendChild(message);

        chatMessages.scrollTop = chatMessages.scrollHeight;

        return message;
    }

    // تحديث حالة الإرسال
    function setSending(value) {
        sending = value;

        if (sendButton) {
            sendButton.disabled = value;
        }

        if (messageInput) {
            messageInput.disabled = value;
        }
    }

    // قراءة JSON بطريقة آمنة
    async function readJSON(response) {
        const responseText = await response.text();

        if (!responseText.trim()) {
            throw new Error(
                `الخادم أعاد استجابة فارغة (HTTP ${response.status}). ` +
                "قد يكون Render أعاد تشغيل الخادم أو انتهت مهلة الطلب."
            );
        }

        let data;

        try {
            data = JSON.parse(responseText);
        } catch (error) {
            console.error(
                "استجابة الخادم ليست JSON صالحًا:",
                responseText.slice(0, 500)
            );

            throw new Error(
                "تعذر قراءة استجابة الخادم. تحقق من سجلات Render."
            );
        }

        if (!response.ok) {
            throw new Error(
                data.error ||
                data.status ||
                `حدث خطأ في الخادم: HTTP ${response.status}`
            );
        }

        return data;
    }

    // فحص حالة النموذج
    async function checkModelStatus() {
        if (!statusElement) {
            return;
        }

        try {
            const response = await fetch("/api/status", {
                method: "GET",
                headers: {
                    "Accept": "application/json"
                },
                cache: "no-store"
            });

            const data = await readJSON(response);

            statusElement.textContent = data.status || (
                data.ready
                    ? "النموذج جاهز"
                    : "النموذج غير جاهز"
            );

            statusElement.classList.toggle(
                "model-ready",
                Boolean(data.ready)
            );

            statusElement.classList.toggle(
                "model-not-ready",
                !data.ready
            );

        } catch (error) {
            console.error("Status error:", error);

            statusElement.textContent =
                "تعذر الاتصال بالنموذج";
        }
    }

    // إرسال الرسالة
    async function sendMessage(text) {
        if (sending || !text.trim()) {
            return;
        }

        const userMessage = text.trim();

        addMessage("user", userMessage);

        conversation.push({
            role: "user",
            content: userMessage
        });

        setSending(true);

        const loadingMessage = addMessage(
            "assistant",
            "يفكر Haroon..."
        );

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify({
                    message: userMessage,
                    history: conversation.slice(-8, -1)
                })
            });

            const data = await readJSON(response);

            if (!data || typeof data.answer !== "string") {
                throw new Error(
                    "وصل رد من الخادم، لكنه لا يحتوي على إجابة صالحة."
                );
            }

            if (loadingMessage) {
                loadingMessage.remove();
            }

            addMessage(
                "assistant",
                data.answer || "لم يتمكن Haroon من توليد إجابة."
            );

            conversation.push({
                role: "assistant",
                content: data.answer
            });

        } catch (error) {
            console.error("Chat error:", error);

            if (loadingMessage) {
                loadingMessage.remove();
            }

            addMessage(
                "assistant",
                `تعذر الحصول على الإجابة: ${error.message}`
            );

            // إزالة رسالة المستخدم من السجل حتى لا يتكرر الطلب
            conversation.pop();

        } finally {
            setSending(false);

            if (messageInput) {
                messageInput.disabled = false;
                messageInput.focus();
            }

            if (sendButton) {
                sendButton.disabled = false;
            }
        }
    }

    // ربط نموذج الدردشة
    if (chatForm && messageInput) {
        chatForm.addEventListener("submit", async (event) => {
            event.preventDefault();

            if (sending) {
                return;
            }

            const text = messageInput.value.trim();

            if (!text) {
                return;
            }

            messageInput.value = "";

            await sendMessage(text);
        });

    } else {
        console.error(
            "لم يتم العثور على عناصر الدردشة. " +
            "تحقق من chatForm وmessageInput في templates/index.html."
        );
    }

    // دعم زر الإرسال إذا كان خارج النموذج
    if (
        sendButton &&
        (!chatForm || !chatForm.contains(sendButton)) &&
        messageInput
    ) {
        sendButton.addEventListener("click", async () => {
            if (sending) {
                return;
            }

            const text = messageInput.value.trim();

            if (!text) {
                return;
            }

            messageInput.value = "";

            await sendMessage(text);
        });
    }

    // فحص حالة النموذج عند فتح الموقع
    checkModelStatus();
});
