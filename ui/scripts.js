const chatBox = document.getElementById("chatBox");
const userInput = document.getElementById("userInput");
const typingIndicator = document.getElementById("typingIndicator");
const sidebar = document.getElementById("sidebar");
const mainContent = document.getElementById("mainContent");
const onboardingScreen = document.getElementById("onboardingScreen");
const overlay = document.getElementById("mobileOverlay");
const memoryList = document.getElementById("memoryList");
const arcContainer = document.getElementById("arcContainer");
const crisisBanner = document.getElementById("crisisBanner");

// --- INITIALIZATION (GATEKEEPER) ---
async function initializeApp() {
    const hasKey = await window.api.hasKey();
    if (hasKey) {
        showMainApp();
    } else {
        onboardingScreen.style.display = "flex";
    }
}

function showMainApp() {
    onboardingScreen.style.display = "none";
    sidebar.style.display = "flex";
    mainContent.style.display = "flex";
}

async function handleKeySubmit() {
    const keyInput = document.getElementById("apiKeyInput");
    const errorMsg = document.getElementById("keyError");
    const key = keyInput.value.trim();

    if (!key.startsWith("AIza")) {
        errorMsg.textContent = "Invalid key format. Gemini keys usually start with 'AIza'.";
        errorMsg.style.display = "block";
        return;
    }

    // Save it securely to the OS via Electron
    const result = await window.api.saveKey(key);
    
    if (result.success) {
        showMainApp();
    } else {
        errorMsg.textContent = "Error saving key securely: " + result.error;
        errorMsg.style.display = "block";
    }
}

// Run init on load
initializeApp();

// --- EXISTING CHAT LOGIC ---

function toggleSidebar() {
    sidebar.classList.toggle("active");
    overlay.classList.toggle("active");
}

function formatEmotionTag(tag) {
    if (!tag) return "";
    return tag.replace('[', '').replace(']', '').replace(/_/g, ' ').toLowerCase();
}

function appendMessage(role, text, tag = null) {
    const div = document.createElement("div");
    div.className = `message ${role}`;

    if (role === 'ai' && tag === '[CRISIS_SIGNAL_ESCALATE]') {
        div.classList.add('crisis-msg');
    }
    
    let content = `<div>${text}</div>`;
    if (tag && tag !== "[NEUTRAL_CONVERSATIONAL]") { 
        content += `<div class="emotion-tag"><span class="dot">●</span> sensing: ${formatEmotionTag(tag)}</div>`;
    }
    
    div.innerHTML = content;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function updateMemoryPanel(entities) {
    if (!entities) return;
    const rows = [];
    for (const [category, data] of Object.entries(entities)) {
        const label = category.replace('context', '').trim();
        if (Array.isArray(data)) {
            data.forEach(item => {
                if (typeof item === 'string' && item.trim()) rows.push({ text: item.trim(), label });
            });
        } else if (typeof data === 'string' && data.trim()) {
            rows.push({ text: data.trim(), label });
        }
    }

    if (rows.length === 0) {
        memoryList.innerHTML = '<li style="color: #BDBAB3; font-style: italic;">Listening and learning...</li>';
        return;
    }

    memoryList.innerHTML = rows
        .map(r => `<li><span>${r.text}</span><span class="entity-type">${r.label}</span></li>`)
        .join("");
}

function updateArcIndicator(historyArray) {
    if (!historyArray || historyArray.length === 0) return;
    let arcHTML = "Trajectory: ";
    const formattedSteps = historyArray.map(tag => `<span class="arc-step">${formatEmotionTag(tag)}</span>`);
    arcHTML += formattedSteps.join(' <span class="arc-arrow">➔</span> ');
    arcContainer.innerHTML = arcHTML;
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    appendMessage("user", text);
    userInput.value = "";
    typingIndicator.style.display = "block";

    try {
        // [IMPORTANT]: We fetch the decrypted key from Electron right before sending!
        const apiKey = await window.api.getKey();

        const data = await window.api.sendToPython('/chat', { 
            message: text, 
            session_id: "default_user",
            api_key: apiKey // Passing the BYOK to the backend
        });

        typingIndicator.style.display = "none";
        appendMessage("ai", data.response, data.emotion_tag);
        
        if (data.emotion_tag === '[CRISIS_SIGNAL_ESCALATE]') {
            crisisBanner.style.display = 'block';
        } else {
            crisisBanner.style.display = 'none';
        }

        if (data.entities) updateMemoryPanel(data.entities);
        if (data.emotion_arc) updateArcIndicator(data.emotion_arc);
        
    } catch (error) {
        typingIndicator.style.display = "none";
        appendMessage("ai", "I seem to have lost my connection. Could you make sure my background process is running?");
        console.error("Bridge Error:", error);
    }
}

function handleEnter(event) {
    if (event.key === "Enter") {
        event.preventDefault(); 
        sendMessage();
    }
}

async function resetMemory() {
    try {
        const res = await window.api.sendToPython('/reset', {});
        chatBox.innerHTML = '<div class="message ai">Memory cleared. Let\'s start fresh whenever you are ready.</div>';
        crisisBanner.style.display = 'none';
        memoryList.innerHTML = '<li style="color: #BDBAB3; font-style: italic;">Listening and learning...</li>';
        arcContainer.innerHTML = 'Trajectory: <span class="arc-step">Establishing Baseline</span>';
        if (window.innerWidth <= 768) toggleSidebar();
    } catch (error) {
        console.error("Reset failed:", error);
        alert("I couldn't reset the memory. Please check the background connection.");
    }
}