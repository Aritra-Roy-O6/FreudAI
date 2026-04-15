const { autoUpdater } = require("electron-updater");
const { app, BrowserWindow, ipcMain, safeStorage, nativeTheme, Tray, Menu, globalShortcut } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;
let pythonProcess;
let tray = null;
const configPath = path.join(app.getPath('userData'), 'freud_config.json');

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    show: false,
    frame: true, // Keep frame for now, but we can go frameless later
    icon: path.join(__dirname, 'assets', 'icon.png'), // Add an icon if you have one
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  mainWindow.loadFile(path.join(__dirname, 'ui', 'index.html'));

  // Prevent app from quitting when window is closed
  mainWindow.on('close', (event) => {
    if (!app.isQuitting) {
      event.preventDefault();
      mainWindow.hide();
    }
    return false;
  });
}

function createTray() {
  tray = new Tray(path.join(__dirname, 'ui', 'favicon.ico')); // Use your logo here
  const contextMenu = Menu.buildFromTemplate([
    { label: 'Show FreudAI', click: () => mainWindow.show() },
    { type: 'separator' },
    { label: 'Quit', click: () => {
        app.isQuitting = true;
        app.quit();
    }}
  ]);
  tray.setToolTip('FreudAI: Your Secure Space');
  tray.setContextMenu(contextMenu);
  tray.on('double-click', () => mainWindow.show());
}

function startPython() {
  const isPackaged = app.isPackaged;
  
  const enginePath = isPackaged 
    ? path.join(process.resourcesPath, 'engine') 
    : path.join(__dirname, 'engine');

  const pythonBin = isPackaged 
    ? path.join(enginePath, 'api.exe') 
    : 'python';

  const pythonArgs = isPackaged ? [] : ['api.py'];

  console.log(`Starting backend at: ${pythonBin}`);

  pythonProcess = spawn(pythonBin, pythonArgs, {
    cwd: enginePath,
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python: ${data.toString()}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    const output = data.toString();
    console.error(`Python Log: ${output}`);

    if (output.includes("Uvicorn running on")) {
      console.log("Backend is ready. Revealing UI...");
      if (mainWindow) {
        mainWindow.show();
        mainWindow.focus();
      }
    }
  });
}

// --- SECURE KEY VAULT (IPC HANDLERS) ---
ipcMain.handle('has-key', () => {
    return fs.existsSync(configPath);
});

ipcMain.handle('save-key', (event, rawKey) => {
    try {
        if (safeStorage.isEncryptionAvailable()) {
            const encryptedBuffer = safeStorage.encryptString(rawKey);
            fs.writeFileSync(configPath, JSON.stringify({ key: encryptedBuffer.toString('base64') }));
            return { success: true };
        } else {
            fs.writeFileSync(configPath, JSON.stringify({ key: rawKey, unencrypted: true }));
            return { success: true };
        }
    } catch (error) {
        console.error("Failed to save key:", error);
        return { success: false, error: error.message };
    }
});

ipcMain.handle('get-key', () => {
    if (!fs.existsSync(configPath)) return null;
    try {
        const data = JSON.parse(fs.readFileSync(configPath));
        if (data.unencrypted) return data.key;
        if (safeStorage.isEncryptionAvailable()) {
            return safeStorage.decryptString(Buffer.from(data.key, 'base64'));
        }
        return null;
    } catch (error) {
        console.error("Failed to decrypt key:", error);
        return null;
    }
});

ipcMain.handle('get-theme', () => {
    if (!fs.existsSync(configPath)) return 'light';
    const config = JSON.parse(fs.readFileSync(configPath));
    return config.theme || 'light';
});

ipcMain.handle('set-theme', (event, theme) => {
    const config = fs.existsSync(configPath) ? JSON.parse(fs.readFileSync(configPath)) : {};
    config.theme = theme;
    fs.writeFileSync(configPath, JSON.stringify(config));
    nativeTheme.themeSource = theme;
    return { success: true };
});

ipcMain.handle('remove-key', () => {
    if (fs.existsSync(configPath)) fs.unlinkSync(configPath);
    return { success: true };
});

app.whenReady().then(() => {
  createWindow();
  createTray();
  startPython();

  // GLOBAL HOTKEY: Alt+Space to summon
  globalShortcut.register('Alt+Space', () => {
    if (mainWindow.isVisible()) {
      mainWindow.hide();
    } else {
      mainWindow.show();
      mainWindow.focus();
    }
  });

  autoUpdater.checkForUpdatesAndNotify();
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
  if (pythonProcess) pythonProcess.kill();
});