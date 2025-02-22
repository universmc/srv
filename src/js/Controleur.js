const { exec } = require('child_process');
const HID = require('node-hid');

const devices = HID.devices();
const vendorId = 2064; // Remplacez par votre vendorId
const productId = 58625; // Remplacez par votre productId

const deviceInfo = devices.find(device => device.vendorId === vendorId && device.productId === productId);

if (deviceInfo) {
    console.log('Contrôleur NES détecté:', deviceInfo);
    const device = new HID.HID(deviceInfo.path);

    device.on('data', (data) => {
        const buttonData = data[5];
        const buttonDataC = data[6];
        const joystickX = data[3]; // Valeur du joystick sur l'axe X
        const joystickY = data[4]; // Valeur du joystick sur l'axe Y


        // Vérifier les boutons
        const buttonSelect = (buttonDataC & 0x10) !== 0; 
        const buttonStart = (buttonDataC & 0x20) !== 0;   
        const buttonA = (buttonData & 0x20) !== 0;        
        const buttonB = (buttonData & 0x10) !== 0;        

        // Mapper les boutons aux actions Makefile
        if (buttonA) runCommand('make bouton_a');
        if (buttonB) runCommand('make bouton_b'); // Bouton B pour Git
        if (buttonStart) runCommand('make bouton_start');
        if (buttonSelect) runCommand('make bouton_select');
        if (joystickX === 0x7f && joystickY === 0x7f) {
            //        console.log('Joystick centré');
        } else if (joystickX < 0x7f) {
            console.log('⬅️ Joystick déplacé vers la gauche ⬅️');
        } else if (joystickX > 0x7f) {
            console.log('Joystick déplacé vers la droite ➡️');
        }

            if (joystickY < 0x7f) {
            console.log('⬆️ Joystick déplacé vers le haut ⬆️');
        } else if (joystickY > 0x7f) {
            console.log('⬇️ Joystick déplacé vers le bas ⬇️');
        }
    });

    device.on('error', (err) => {
        console.error('Erreur de périphérique:', err);
    });

} else {
    console.log('Contrôleur NES non trouvé');
}

// Fonction pour exécuter une commande
function runCommand(command) {
    exec(command, (err, stdout, stderr) => {
        if (err) {
            console.error(`Erreur lors de l'exécution de la commande ${command}:`, err);
            return;
        }
        if (stderr) {
            console.error(`Erreur de commande: ${stderr}`);
            return;
        }
        console.log(`Résultat de la commande: ${stdout}`);
    });
}
