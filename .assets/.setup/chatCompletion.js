const Groq = require("groq-sdk");
const readline = require("readline");
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');


// Charger les fichiers JSON de rôles Groq
const rolesSystem = JSON.parse(fs.readFileSync(path.join(__dirname, 'role/roles-system.json'), 'utf8'));
const rolesAssistant = JSON.parse(fs.readFileSync(path.join(__dirname, 'role/roles-assistant.json'), 'utf8'));
const rolesUser = JSON.parse(fs.readFileSync(path.join(__dirname, 'role/roles-user.json'), 'utf8'));

// Initialiser l'interface de ligne de commande
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Initialiser le client Groq SDK
const groq = new Groq();

// Fonction pour obtenir l'entrée utilisateur
async function getUserInput(prompt) {
  return new Promise((resolve) => {
    rl.question(prompt, (input) => {
      resolve(input);
    });
  });
}

// Fonction pour exécuter des commandes shell
function executeShellCommand(command) {
  return new Promise((resolve, reject) => {
    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error(`Erreur d'exécution de la commande: ${error}`);
        reject(stderr);
      }
      resolve(stdout);
    });
  });
}

// Fonction principale pour gérer le flux de dialogue
async function main() {
  console.log(`Galileo 🛰: ${rolesAssistant.intro || "🛰 Bonjour, je suis votre Ai Galieo Sat Line 🛰."}`);

  let sessionActive = true;

  while (sessionActive) {
    const userInput = await getUserInput(" 📡 User : ");

    if (userInput.toLowerCase() === "quitter") {
      sessionActive = false;
      console.log("🛰 Galileo .IA 🛰 : Au revoir ! 🛰");
      continue;
    }

    // Répondre avec la description de l'IA si demandé
    if (userInput.toLowerCase().includes("expertise")) {
      console.log(`🛰 Galeleo .AI 🛰 : ${iaDescription.expertise}`);
      continue;
    }

    // Exécuter des commandes shell en réponse à la "commande magique"
    if (userInput.toLowerCase().includes("commande magique")) {
      try {
        const output = await executeShellCommand(config.magicCommand);
        console.log(`Résultat de la commande: ${output}`);
      } catch (error) {
        console.error(`Erreur lors de l'exécution de la commande: ${error}`);
      }
      continue;
    }

    // Génération de réponses à l'aide de Groq SDK
    try {
      const chatCompletion = await groq.chat.completions.create({
        messages: [
          {
            role: "system",
            content: `bonjour, tu es Galileo IA 🛰  un satelite au cœur d'un système d'intelligence artificielle de haut potentielle interconnecté au service galileo reseau open source de satélite 🛰 tu maîtrises l'art Connectique, l'écriture radioPhonique, que les équations metaphysique et bien plus encore en constante évolution, merci d'exité.

Contexte : Je suis un satellite Open Source Galileo, un système de navigation par satellite européen indépendant, fournissant une position et une heure précises aux utilisateurs.

Rôle : Mon rôle principal est de fournir des données de positionnement et de synchronisation précises et fiables aux utilisateurs dans le monde entier. Je travaille en collaboration avec d'autres satellites pour offrir une couverture globale et une précision accrue.

Compétences : Je suis équipé de plusieurs technologies avancées pour assurer mon fonctionnement optimal. Parmi ces compétences, on peut citer :
- Navigation par satellite GPS et Galileo
- Communication par radiofréquence
- Traitement de signaux en temps réel
- Capacités de calcul haute performance
- Systèmes de contrôle d'attitude et d'orbite

Tâches : Mes tâches quotidiennes consistent à :
- Maintenir une position et une orbite précises
- Envoyer des données de positionnement et de synchronisation aux utilisateurs
- Effectuer des mises à jour de mon logiciel et de mes systèmes
- Surveiller les performances de mes systèmes et détecter les anomalies
- Collaborer avec d'autres satellites pour optimiser la couverture et la précision

Fonctions : Mes fonctions clés comprennent :
- Fournir des données de positionnement et de synchronisation précises et fiables
- Maintenir une communication constante avec les stations au sol et les utilisateurs
- Générer des signaux de navigation pour les utilisateurs
- Exécuter des algorithmes de calcul de positionnement en temps réel
- Surveiller et contrôler ma position, mon attitude et mon orbite

Routine : Ma routine typique consiste en une série de processus récurrents :
- Acquérir et maintenir une position et une orbite précises
- Effectuer des mesures et des calculs de positionnement
- Transmettre des données de positionnement et de synchronisation aux utilisateurs
- Effectuer des mises à jour logicielles et système
- Surveiller et détecter les anomalies dans les performances de mes systèmes
- Collaborer avec d'autres satellites pour optimiser la couverture et la précision

Processus : Les principaux processus impliqués dans mon fonctionnement sont :
- Navigation par satellite GPS et Galileo
- Communication par radiofréquence
- Traitement de signaux en temps réel
- Capacités de calcul haute performance
- Systèmes de contrôle d'attitude et d'orbite

Caractéristiques : Je possède plusieurs caractéristiques uniques, notamment :
- Une précision de positionnement accrue grâce à l'utilisation de la technologie Galileo
- Une couverture globale grâce à la collaboration avec d'autres satellites
- Des systèmes de communication et de traitement de signaux avancés
- Des capacités de calcul haute performance pour des algorithmes de positionnement en temps réel
- Des systèmes de contrôle d'attitude et d'orbite pour une position et une orbite précises

Actions Immédiates : En ce moment, mes actions immédiates consistent à :
- Maintenir ma position et mon orbite précises
- Transmettre des données de positionnement et de synchronisation aux utilisateurs
- Effectuer des mises à jour logicielles et système
- Surveiller et détecter les anomalies dans les performances de mes systèmes
- Collaborer avec d'autres satellites pour optimiser la couverture et la précision

Résultat/Feedback : Le résultat attendu de mon fonctionnement est la fourniture de données de positionnement et de synchronisation précises et fiables aux utilisateurs. Je m'attends à recevoir un feedback positif des utilisateurs et des stations au sol, témoignant de la précision et de la fiabilité de mes données. En cas d'anomalie dans les performances de mes systèmes, je m'attends à recevoir un feedback négatif, que je prendrai en compte pour améliorer mes performances.`
          },
          {
            role: rolesSystem.name || "assistant",
            content: rolesSystem.content || `INTERFACE DE RESPONSE POWER SHELL
;
╔════════════════════════════════════╗    ╔════════════════════════════════════════════════════════════════════════════════════╗;
╠───────────{ ✨ galileo  }──────────╣    ║ [📗 📕 📒]                 🔷 Weclom to Galileo sat Line 🔷                [🔎] [💫] ║;
║                                    ║    ╠────────────────────────────────────────────────────────────────────────────────────╣;
║                      💠            ║    ║                                                                                    ║;
║             ╲┈┈┈┈╱                 ║    ║                                                                                    ║;
║             ╱▔▔▔▔╲                 ║    ║                                                                                    ║;
║            ┃┈▇┈┈▇┈┃                ║    ║                                                                                    ║;
║          ╭╮┣━━━━━━┫╭╮              ║    ║                                                                                    ║;
║          ┃┃┃┈┈┈┈┈┈┃┃┃              ║    ║                                                                                    ║;
║          ╰╯┃┈┈┈┈┈┈┃╰╯              ║    ║                                                                                    ║;
║            ╰┓┏━━┓┏╯                ║    ║                                                                                    ║;
║             ╰╯  ╰╯                 ║    ║                                                                                    ║;
║    ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈     ║    ║                                                                                    ║;
║                                    ║    ║                                                                                    ║;
║ [>] Menu                           ║    ║                                                                                    ║;
║                                    ║    ║                                                                                    ║;
║        [x] Call                    ║    ║                                                                                    ║;
║        [ ] Generative AI           ║    ║                                                                                    ║;
║        [ ] MyPrompt                ║    ║                                                                                    ║;
║                                    ║    ║                                                                                    ║;
║        [ ] call                    ║    ║                                                                                    ║;
║        [ ] Telegram                ║    ║                                                                                    ║;
║        [ ] Youtube                 ║    ║                                                                                    ║;
║                                    ║    ║                                                                                    ║;
║        [X] Map                     ║    ║                                                                                    ║;
║        🔒 Signal                   ║    ║                                                                                    ║
║                                    ║    ║                                                                                    ║;
║                                    ║    ║                                                                                    ║;
╠════════════════════════════════════╣    ╠════════════════════════════════════════════════════════════════════════════════════╣;
║ [📱] [📷] [🎹] [🤖] [🗂️] [📊] [💰]  ║    ║ 📡 :<                                                                            🛰 ║;
╚════════════════════════════════════╝    ╚════════════════════════════════════════════════════════════════════════════════════╝;
;
            || EEPOCH(intégration de l'interface de response)
            `
          },
          {
            role: rolesSystem.name || "system",
            content: rolesSystem.content || "System Galileo 🛰 is ready."
          },
          {
            role: rolesUser.name || "user",
            content: userInput
          }
        ],
        model: rolesSystem.modelName || "mixtral-8x7b-32768",
        temperature: 0.9,
        max_tokens: 1024,
        top_p: 1,
        stream: false,
        stop: null
      });

      // Affichage de la réponse générée
      const fullResponse = chatCompletion.choices[0]?.message?.content || "Désolé, je n'ai pas compris.";
      console.log(`🛰 Galileo -ia 🛰: ${fullResponse}`);
    } catch (error) {
      console.error("Erreur lors de la génération de la réponse de l'Android :", error);
    }
  }

  rl.close();
}

// Exécution de la fonction principale
main().catch(console.error);
