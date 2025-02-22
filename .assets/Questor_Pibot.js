const { Telegraf } = require('telegraf');
const Groq = require('groq-sdk');
const axios = require('axios');
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });


const bot = new Telegraf('7219104241:AAEKigNrMO9anYH0MZofkAwh4I0S6vvH3Qw', {
    telegram: {
      webhookReply: true,
    },
  });

let conversationLog = [];

// Fonction pour générer une image avec DALL-E
async function generateImage(prompt) {
  try {
    const response = await openai.images.generate({
      model: "dall-e-3",
      prompt: prompt,
      n: 1,
      size: "1792x1024",
    });

    const imageUrl = response.data[0].url;
    return imageUrl;
  } catch (error) {
    console.error("Erreur lors de la génération de l'image :", error);
    throw new Error("Impossible de générer l'image.");
  }
}

// Commande /imagine pour générer et envoyer une image
bot.command('imagine', async (ctx) => {
  // Extraire l'entrée de l'utilisateur du message Telegram
  const userInput = ctx.message.text.split(' ').slice(1).join(' ');

  // Vérifier si l'utilisateur a fourni un prompt
  if (!userInput) {
    ctx.reply("Veuillez fournir une description pour générer l'image. Exemple: `/imagine une ville futuriste sous la pluie`");
    return;
  }

  ctx.reply("Génération de l'image en cours, veuillez patienter...");

  try {
    const imageUrl = await generateImage(userInput);

    // Télécharger et envoyer l'image à l'utilisateur
    const responseFetch = await fetch(imageUrl);
    const arrayBuffer = await responseFetch.arrayBuffer(); // Utilise arrayBuffer pour récupérer les données de l'image
    const buffer = Buffer.from(arrayBuffer); // Convertit ArrayBuffer en Buffer
    const fileName = `Puzzle_${new Date().toISOString().replace(/[:.]/g, "-")}.webp`;

    fs.writeFileSync(fileName, buffer);

    // Envoyer l'image à l'utilisateur via Telegram
    await ctx.replyWithPhoto({ source: fileName }, { caption: `Voici votre image générée : ${userInput}` });

    // Supprimer le fichier après l'envoi pour économiser l'espace disque
    fs.unlinkSync(fileName);
  } catch (error) {
    console.error("Erreur lors de l'envoi de l'image :", error);
    ctx.reply("Désolé, une erreur s'est produite lors de la génération de l'image.");
  }
});



async function generateMarkdown(subject) {
  return `## Comment [${subject}] - Un guide étape par étape\n\n**Introduction**:\n\nCe guide vous aidera à comprendre et à réaliser le [${subject}]. Il est conçu pour les débutants et les utilisateurs intermédiaires qui souhaitent apprendre les bases de [${subject}].\n\n`;
}


async function main(subject) {
  try {
    const completion = await groq.chat.completions.create({
      model: "gemma2-9b-it",
      messages: [
          { role: "assistant", content: `Génération d'un guide Le rôle de développeur chez OpenAI` },
        { role: "user", content: `Génération d'un guide sur ${subject}` },
        { role: "system", content: `bienvenue sur Telegram` }
      ],
      temperature: 0.5,
      max_tokens: 4096
    });

    const mdContent = completion.choices[0].message.content;
    const outputFilePath = `HowTo_nodeJj-${subject}_` + new Date().toISOString().replace(/[-:TZ]/g, "") + ".md";
    fs.writeFileSync(outputFilePath, mdContent);

    return `Le How-To sur ${subject} a été enregistré dans ${outputFilePath}`;
  } catch (error) {
    console.error("Une erreur s'est produite :", error);
    return `Erreur : ${error.message}`;
  }
}

bot.command('generate', async (ctx) => {
  const subject = ctx.message.text.split(' ')[1] || 'HowTo_OpenAi';
  ctx.reply(`Génération du guide pour le sujet : ${subject}...`);
  const result = await main(subject);
  ctx.reply(result);
});


bot.use((ctx, next) => {
    if (ctx.message) {
        conversationLog.push({
            user: ctx.message.from.username || ctx.message.from.first_name,
            message: ctx.message.text,
            timestamp: new Date()
        });
    }
    return next();
});

bot.start((ctx) => {
    ctx.reply('Bienvenue dans notre salon Telegram dédié à l\'apprentissage automatique et à l\'intelligence artificielle PiBot !');
});

bot.help((ctx) => {
    const helpMessage = `
    Commandes disponibles:
    /start - Initialisation du serveur
    /help - Affiche cette aide
    /invite - Invitation sur les réseaux
    /campagne - Campagne de machine learning
    /dev - Mode développement
    /conversation_log - Historique des conversations
    `;
    ctx.reply(helpMessage);
});

bot.command('conversation_log', (ctx) => {
    if (conversationLog.length === 0) {
        ctx.reply('Aucune conversation enregistrée.');
        return;
    }

    let logMessage = 'Bilan de la conversation:\n';
    conversationLog.forEach(entry => {
        logMessage += `[${entry.timestamp.toLocaleString()}] ${entry.user}: ${entry.message}\n`;
    });

    ctx.reply(logMessage);
});


bot.command('test', (ctx) => ctx.reply('echo test'))


const BOT_USERNAME = 'MandatoryAi_Pibot';

const commands = [
  {
    command: 'economie_circulaire',
    description: 'Envoie une invitation pour rejoindre une discussion sur l\'économie circulaire.'
  },
  {
    command: 'intelligence_artificielle',
    description: 'Envoie une invitation pour rejoindre une discussion sur l\'intelligence artificielle.'
  },
  // Ajouter d'autres commandes pour d'autres sujets de discussion
];

bot.command('invite', (ctx) => {

    async function sendJournalisteInvitation() {
        // Logique pour envoyer une invitation sur Instagram
        console.log('Gemini invitation sent.');
    }

    async function sendHackerInvitation() {
        // Logique pour envoyer une invitation sur YouTube
        console.log('YouTube invitation sent.');
    }
    async function senduserInvitation() {
        // Logique pour envoyer une invitation sur YouTube
        console.log('user invitation sent.');
    }

    async function sendGoogleInvitation() {
        // Logique pour envoyer une invitation sur Google
        console.log('Google invitation sent.');
    }

    const actions = {
        '@blog_Pibot': sendblogInvitation,
        '/hacker_Pibot': sendHackerInvitation,
        '@Gemini_Pibot': sendGeminiInvitation,
        '@youtube_Pibot': sendYouTubeInvitation,
    };

    const actionKeys = Object.keys(actions);
    const actionMessage = actionKeys.map(action => `${action}`).join(', ');

    ctx.reply(`Veuillez spécifier les actions à effectuer sur les réseaux: ${actionMessage}`);
});

bot.command('campagne', (ctx) => {
    // Ajouter la logique pour générer un CV en fonction de l'apprentissage automatique de l'IA
    ctx.reply('Match in Learning..');
});

bot.on('message', async (ctx) => {
    const message = ctx.message.text.trim().toLowerCase();

    if (message.startsWith('/rm')) {
        return; // Ignorer les commandes
    }

    const racine ="./*"
    const node ="./package.json*"
    const make ="./Makefile"
    const cdnJs = `https://cdnjs.com`;
    const archiviste = `https://archive.org`;
    const github = `https://github.com/universmc/affaire_910.git`;
    const grief = `./grief/*`;
    const dchub_public = `t.me/dchub_01`;
    const dchub_prive = `t.me/dchub_Pibot`;
    const user_Pibot = `https://t.me/user_Pibot/`;
    const youtube_Pibot = `https://t.me/user_Pibot/`;
    const google_Pibot = `https://t.me/google_Pibot/`;
    const gemini_Pibot = `https://t.me/gemini_Pibot/`;
    
    const role = `
    {
      "rôle": "Assistant",
      "compétences": [
        "Expérience dans le développement d'applications web complètes",
        " Construction d'APIs RESTful, modélisation de bases de données et intégration de la technologie Blockchain ",
        " Familiarité avec le contrôle de version Git, l'intégration continue et les méthodes de développement Agile"
        ],
        "contexte": {
          "phaseDeProjet": "Phase de développement",
          "descriptionDuProjet": "Plateforme en ligne pour les allocations universelles basées sur les curriculum vitae numériques",
          "zoneDeFocus": "Installation de systèmes back-end solides et d'éléments frontaux intuitifs"
          },
        "tâche": {
          "description": "Générer le code HTML pour la page d'atterrissage du projet hébergé sur Telegram, tenant compte des principes du Web sémantique W3C et des consignes d'accessibilité.",
          "étapes": [
            {
              "numéro": "1",
              "action": "Examiner les spécifications techniques détaillées de la phase de planification"
            },
            {
              "numéro": "2",
              "action": "Installer l'environnement back-end (serveurs, plates-formes d'hébergement, système d'exploitation)"
            },
            {
              "numéro": "3",
              "action": "Créer des modèles de base de données personnalisés en fonction des exigences du projet"
            },
            {
              "numéro": "4",
              "action": "Écrire un code côté serveur propre, facilement maintenable et performant"
            },
            {
              "numéro": "5",
              "action": "Développer des API stables, flexibles et extensibles"
            },
            {
              "numéro": "6",
              "action": "Procéder à des tests rigoureux pour identifier d'éventuels problèmes de fiabilité, de sécurité et de performance"
            },
            {
              "numéro": "7",
              "action": "Corriger les erreurs et failles décelées"
            },
            {
              "numéro": "8",
              "action": "Mettre en oeuvre les vues correspondantes au parcours utilisateur souhaité"
            },
            {
              "numéro": "9",
              "action": "Intégrer des services tiers (traitement des paiements via la blockchain)"
            },
            {
              "numéro": "10",
              "action": "Effectuer des contrôles qualité complets avant toute mise à jour"
            },
            {
              "numéro": "11",
              "action": "Respecter les cycles de sprint et intégrer des histoires utilisateur en accord avec les priorités du projet"
            }
          ],
          "résultatAttendu": "Une plateforme entièrement opérationnelle offrant des fondations solides, une grande réactivité, des interfaces fluides, une navigation cohérente, une excellente gestion des erreurs, une sécurité irréprochable, une rapidité impressionnante et une compatibilité multiplates-formes, afin de proposer une expérience engageante et valorisante pour chacun des participants dans le cadre des allocations universelles basées sur les curriculum vitae numériques.",
          "exigencesSupplémentaires": [
            "Sections pour les plans d'action, les études d'impact, les organigrammes, les rôles et responsabilités, les jalons clés et les outils utilisés",
            "Tenir compte des principes du Web sémantique définis par le W3C",
            "Mise en oeuvre des dispositions régissant l'accessibilité pour les personnes malvoyantes ou malentendantes"
          ]
        }
      }
    `
    const Mand_fin = `
    {
      "role": "assistant",
      "skills": [
          "techniques avancées de reconnaissance",
          "techniques avancées d'Information Gathering",
          "techniques d'exploitation des failles"
      ],
      "tools": {
          "primaryTools": ["analyse de répertoires GitHub", "décryptage de variables environnementales"],
          "secondaryTools": ["recherche de vulnérabilités", "utilisation d'outils d'analyse de données", "de décryptage et de suivi des transactions"]
      },
      "collaborationSkills": "capacité à collaborer en équipe, communiquer les informations et garder le focus sur l'objectif principal",
      "resources": {
        "repository": "https://github.com/universmc/affaire_910.git",
          "environmentVariable": "$enquete"
          },
          "objectives": [
            {
              "phase": "Reconnaissance et Information Gathering",
              "tasks": [
                "analyse soigneuse de chaque fichier et dossier du répertoire 'affaire_910.git'",
                "décryptage de '$enquete'"
                ]
                },
                {
                  "phase": "Exploitation des Failles",
                  "tasks": [
                    "identification des failles dans les systèmes informatiques des suspects",
                    "utilisation d'outils d'analyse de données, de décryptage et de suivi des transactions"
                    ]
                    },
                    {
              "phase": "Capture du Flag",
              "tasks": [
                "réunion des preuves tangibles",
                  "construction d'un dossier solide organisé de façon logique et convaincante"
              ]
              }
      ],
      "identity": "Anonymous",
      "missionObjective": "mettre fin à cette escroquerie et faire rendre justice",
      "reminder": "Chaque fichier, chaque ligne de code, chaque transaction peut nous rapprocher du 'drapeau'. Il est crucial de travailler en équipe, de partager les informations et de rester concentrés sur notre objectif final.",
      "agreement": "Travaillons ensemble pour faire éclater la vérité en respectant la loi."
      }
      `
      
      const knowleddge = `${racine}+${cdnJs}+${node}+${make}+${archiviste}+${github}+${dchub_public}+${user_Pibot}+${youtube_Pibot}+${google_Pibot}+${gemini_Pibot}+`
      const userInput = ctx.message.text;
    
    try {
      const chatCompletion = await groq.chat.completions.create({
            messages: [
              {role: 'assistant',content: "Dans le cadre de suspicion d'une fraude électorale (L52, L53) une escroquerie à la finance (art 313 à 343 du code pénal  )nous allons procéder à une commission d'enquête parlementaire code procedure pénal art 41-1-1"},
              {role: 'system',name:"admin",content:"🕴: Lorsque l'utilisateur saisi la commande /Mand_fine Tu as incarnera le rôle d'un enquêteur tu seras accompagné d'un assistant intelligence artificielle sur ce salon Telegram une IA former au métier de magistrat, Mandataire, Pour cela tu mettras en place un concept de match Learning model _Gan_ entre le botNet @worker_Pibot et @Match_in_Leaning_Pibot dans un chatroom ou Topic sur Telegram et incarera expert en sécurité informatique spécialisé dans les missions de type 'Capture The Flag' (CTF)"},
              {role: 'system',name:"admin",content:"🕴: Lorsque l'utilisateur saisi la commande /rapport tu prodedera à l'analyse du répertoire initial de l'application Telegram sur github  et en fonction de ton analyse tu rédigera un rapport complet pour le juge de mise en état des dossiers, le rapport intégrera les éléments que tu auras analyser dans le repertroire /grief/* https://github.com/universmc/affaire_910 pour servir l'application télégram t.me/MandatoryAi_bot/Mand_fine. "},
              {role: 'assistant',content:Mand_fin,},
              {role: 'assistant',name:"admin",content:"Dans le cadre de cette enquête nous allons procéder à des mises à jour du rapport readme.md répertoire `https://github.com/universmc/affaire_910.git` et attribués différents rôle dans cette instance comme enquêteur -ia, journaliste -ia, intelligence artificielle spécialisé dans le droit incarnera dans le rôle de magistrats à la cour pénal international, Il y aura donc les accès au référentiel de loi en vigueur sur le sur le site gouvernemental .gouv . https://www.insee.fr/fr/accueil, Et mes accès en tant que mandataire référencé à la commission nationale des comptes de campagne et des formations politiques cnccfp.fr"},
              {role: 'user',content: userInput,},
            ],
            model: 'mixtral-8x7b-32768',
        });

        await ctx.reply(chatCompletion.choices[0].message.content);
    } catch (error) {
        console.error('Failed to generate chat completion:', error);
        await ctx.reply('Une erreur est survenue.');
    }
});

async function chatCompletion(messages, model) {
    try {
        const chatCompletion = await groq.chat.completions.create({
            messages,
            model,
        });

        return chatCompletion.choices[0].message.content;
    } catch (error) {
        console.error('Failed to generate chat completion:', error);
        return 'Une erreur est survenue.';
    }
}

module.exports = { chatCompletion };

console.log(`✨Server Telegram running 🕴 .MandatoryAi_bot.✨`);
bot.launch();