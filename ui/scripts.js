// --- DOM ELEMENTS ---
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
const settingsModal = document.getElementById("settingsModal");

// --- INITIALIZATION ---
async function initializeApp() {
    const theme = await window.api.getTheme();
    document.getElementById("themeSelect").value = theme;
    applyTheme(theme);
    
    if (await window.api.hasKey()) { 
        showMainApp(); 
    } else { 
        onboardingScreen.style.display = "flex"; 
    }
}

function applyTheme(theme) {
    theme === 'dark' ? document.body.classList.add('dark-mode') : document.body.classList.remove('dark-mode');
}

async function updateTheme() {
    const theme = document.getElementById("themeSelect").value;
    await window.api.setTheme(theme);
    applyTheme(theme);
}

function showMainApp() {
    onboardingScreen.style.display = "none";
    sidebar.style.display = "flex";
    mainContent.style.display = "flex";
}

// --- GATEKEEPER / AUTH ---
async function handleKeySubmit() {
    const key = document.getElementById("apiKeyInput").value.trim();
    if (!key.startsWith("AIza")) {
        const err = document.getElementById("keyError");
        err.textContent = "Invalid format. Gemini keys usually start with 'AIza'.";
        err.style.display = "block";
        return;
    }
    const result = await window.api.saveKey(key);
    result.success ? showMainApp() : alert("Secure storage failed: " + result.error);
}

// --- SETTINGS ---
async function toggleSettings() {
    if (settingsModal.style.display === "flex") {
        settingsModal.style.display = "none";
    } else {
        settingsModal.style.display = "flex";
        document.getElementById("settingsKeyInput").value = await window.api.getKey() || "";
    }
}

function toggleKeyVisibility() {
    const input = document.getElementById("settingsKeyInput");
    input.type = input.type === "password" ? "text" : "password";
}

async function removeKey() {
    if (confirm("Remove API Key and lock the app?")) { 
        await window.api.removeKey(); 
        window.location.reload(); 
    }
}

// --- CHAT LOGIC ---
function appendMessage(role, text, tag = null) {
    const div = document.createElement("div");
    div.className = `message ${role}`;
    let content = `<div>${text}</div>`;
    
    if (tag && tag !== "[NEUTRAL_CONVERSATIONAL]") {
        const cleanTag = tag.replace(/[\[\]]/g, '').replace(/_/g, ' ').toLowerCase();
        content += `<div class="emotion-tag">sensing: ${cleanTag}</div>`;
    }
    
    div.innerHTML = content;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    appendMessage("user", text);
    userInput.value = "";
    typingIndicator.style.display = "block";

    try {
        const apiKey = await window.api.getKey();
        const data = await window.api.sendToPython('/chat', { 
            message: text, 
            api_key: apiKey 
        });

        typingIndicator.style.display = "none";

        // PHASE 4: ERROR INTERCEPTOR HANDLING
        if (data.error) {
            if (data.error_type === "invalid_key") {
                appendMessage("ai", "⚠️ Your API key is invalid. Please check your credentials in settings.");
                toggleSettings(); // Force settings open
            } else if (data.error_type === "quota_exceeded") {
                appendMessage("ai", "⚠️ Google AI Studio quota exceeded. Please wait or check your billing tier.");
            } else {
                appendMessage("ai", "⚠️ Engine Error: " + data.message);
            }
            return;
        }

        appendMessage("ai", data.response, data.emotion_tag);
        
        // Crisis check
        crisisBanner.style.display = data.emotion_tag === '[CRISIS_SIGNAL_ESCALATE]' ? 'block' : 'none';
        
        // Update Side Panels
        if (data.entities) updateMemoryPanel(data.entities);
        if (data.emotion_arc) updateArcIndicator(data.emotion_arc);

    } catch (e) {
        typingIndicator.style.display = "none";
        appendMessage("ai", "I've lost connection to my background engine.");
    }
}

// --- MEMORY / AGENCY LOGIC ---
function updateMemoryPanel(entities) {
    if (!entities) return;
    const rows = [];
    Object.entries(entities).forEach(([cat, data]) => {
        if (Array.isArray(data)) data.forEach(item => rows.push({ text: item, label: cat }));
    });

    if (rows.length === 0) { 
        memoryList.innerHTML = '<li>Listening...</li>'; 
        return; 
    }

    memoryList.innerHTML = rows.map(r => `
        <li class="memory-item" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <div style="flex:1;">
                <span>${r.text}</span>
                <span style="display:block; font-size:10px; opacity:0.6;">${r.label.replace('context', '')}</span>
            </div>
            <button onclick="forgetEntity('${r.label}', '${r.text.replace(/'/g, "\\'")}')" 
                    style="color:#ff4d4d; background:none; border:none; cursor:pointer; font-size:16px;">×</button>
        </li>
    `).join("");
}

async function forgetEntity(category, item) {
    try {
        const data = await window.api.sendToPython('/forget-entity', { category, item });
        updateMemoryPanel(data.updated_entities);
    } catch (e) {
        console.error("Forget failed", e);
    }
}

function updateArcIndicator(historyArray) {
    if (!historyArray) return;
    arcContainer.innerHTML = "Trajectory: " + historyArray.map(t => `<span class="arc-step">${t.replace(/[\[\]]/g, '').toLowerCase()}</span>`).join(" ➔ ");
}

function handleEnter(e) { if (e.key === "Enter") sendMessage(); }

async function resetMemory() {
    if (confirm("Reset all memories?")) {
        await window.api.sendToPython('/reset', {});
        window.location.reload();
    }
}

// Start
initializeApp();