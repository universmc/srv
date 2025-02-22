const { app, Menu, MenuItem, BrowserWindow } = require('electron');

function createMenu() {
  const template = [
    {
        label: 'Electra',
        submenu: [
          { label: 'Nouveau' },
          { label: 'Ouvrir' },
          { type: 'separator' },
          { label: 'Quitter', role: 'quit' }
        ]
      },
      {
        label: 'Édition',
        submenu: [
          { label: 'Annuler', role: 'undo' },
          { label: 'Rétablir', role: 'redo' },
          { type: 'separator' },
          { label: 'Couper', role: 'cut' },
          { label: 'Copier', role: 'copy' },
          { label: 'Coller', role: 'paste' },
          { label: 'Supprimer', role: 'delete' },
          { label: 'Sélectionner tout', role: 'selectall' }
        ]
      },
      {
        label: 'Models',
        submenu: [
          {
            label: 'Groq',
            click: () => {
              // Affichez une boîte de dialogue ou une fenêtre avec les informations "À propos"
              const aboutWindow = new BrowserWindow({ /* ... */ });
              aboutWindow.loadFile('about.html');
            }
          },
          {
            label: 'Gemini',
            click: () => {
              // Affichez une boîte de dialogue ou une fenêtre avec les informations "À propos"
              const aboutWindow = new BrowserWindow({ /* ... */ });
              aboutWindow.loadFile('about.html');
            }
          },
          {
            label: 'ollama',
            click: () => {
              // Affichez une boîte de dialogue ou une fenêtre avec les informations "À propos"
              const aboutWindow = new BrowserWindow({ /* ... */ });
              aboutWindow.loadFile('about.html');
            }
          },
          {
            label: 'DeepSeek',
            click: () => {
              // Affichez une boîte de dialogue ou une fenêtre avec les informations "À propos"
              const aboutWindow = new BrowserWindow({ /* ... */ });
              aboutWindow.loadFile('about.html');
            }
          },
          {
            label: 'Avatars',
            click: () => {
              // Affichez une boîte de dialogue ou une fenêtre avec les informations "À propos"
              const aboutWindow = new BrowserWindow({ /* ... */ });
              aboutWindow.loadFile('about.html');
            }
          },
          {
            label: 'GPT',
            click: () => {
              // Affichez une boîte de dialogue ou une fenêtre avec les informations "À propos"
              const aboutWindow = new BrowserWindow({ /* ... */ });
              aboutWindow.loadFile('about.html');
            }
          }
        ]
      },
      {
        label: 'Blog',
        submenu: [
          {
            label: 'Blog_DevOps',
            click: () => {
              // Affichez une boîte de dialogue ou une fenêtre avec les informations "À propos"
              const aboutWindow = new BrowserWindow({ /* ... */ });
              aboutWindow.loadFile('about.html');
            }
          },
          {
            label: 'Kjournal',
            click: () => {
              // Affichez une boîte de dialogue ou une fenêtre avec les informations "À propos"
              const aboutWindow = new BrowserWindow({ /* ... */ });
              aboutWindow.loadFile('about.html');
            }
          }
        ]
      },
      {
        label: 'Affichage',
        submenu: [
          { label: 'Recharger', role: 'reload' },
          { label: 'Forcer le rechargement', role: 'forcereload' },
          { label: 'Outils de développement', role: 'toggledevtools' },
          { type: 'separator' },
          { label: 'Zoom +', role: 'zoomin' },
          { label: 'Zoom -', role: 'zoomout' },
          { label: 'Réinitialiser le zoom', role: 'resetzoom' }
        ]
      },
      {
        label: 'Aide',
        submenu: [
          {
            label: 'À propos',
            click: () => {
              // Affichez une boîte de dialogue ou une fenêtre avec les informations "À propos"
              const aboutWindow = new BrowserWindow({ /* ... */ });
              aboutWindow.loadFile('about.html');
            }
          }
        ]
      }
    ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

module.exports = { createMenu };