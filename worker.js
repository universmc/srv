const { Telegraf } = require('telegraf');
const Groq = require('groq-sdk');
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });


const bot = new Telegraf('apiKey', {
  telegram: {
    webhookReply: true,
  },
});

const BOT_USERNAME = 'Pibot';

const config = {
    "projet" : {
      "name":"univers-mc",
      "admin":"Mickael",
      "root":"@Pi-ia-bot",
      "DirectLink": "https://t.me/Pi_ia_Pibot/news",
      "siteWeb":"https://ia.univers-mc.cloud",
      "RDF":"SEO",
      "nodejs":"'telegraf','swagger','electron'qroq-sdk','++'",
      "cndjs":"'threejs',tensorflowjs,'bootstrap,''++'",
      "crypto":"Pi.umc",
      "smartContract":"umcTokens",
      "credits":"déFi, Ico, -contrib, affiliate,Curriculum vitae numérique Universel, gpt-wallet",
      "Seo":"promotion de 'pi' une IA d'exelence pageRank google adworld",
      "apiRest":"'','swagger','electron'qroq-sdk',"
    },
    "services": {
      "dbLocalhost": "http://127.0.0.1:3010/",
      "ports": {
        "php": 8080,
        "mysql": 3306,
        "ftp": 21,
        "ssh": 22,
        "nodejs": 3000
      },
      "ftp": {
        "host": "ftp.cluster027.hosting.ovh.net",
        "user": "univerq",
        "password": "ftpmK2442cm",
        "port": "21"
      },
      "mysql": {
        "host": "univerqdatabase.mysql.db",
        "database": "univerqdatabase",
        "user": "univerqdatabase",
        "password": "sqlmK2442cm"
    }
    },
    "Telegram": {
      "blog": "@dchub_blog",
      "News": "https://ia.univers-mc.cloud/",
      "hub_Pibot": "https://github.com/universmc/zira",
      "hub_public": "https://github.com/universmc/pi",
      "url": "https://github.com/universmc/",
      "youtube_studio" : "https://t.me/youtube_Pibot/",
      "google_workspace" : "https://t.me/google_Pibot/",
      "studio" : "https://t.me/studio_Pibot/",
      "Avatars" : "https://t.me/Avatars_Pibot/",
      "hub_prive" : "https://t.me/dchub_Pibot",
      "gemini" : "https://t.me/gemini_Pibot/",
      "user_Pibot" : "https://t.me/user_Pibot",
      "pi" : "https://t.me/pi_Pibot/",
      "match_in_learning" : "https://t.me/Match_in_Leaning_Pibot"
    },
    "github": {
      "url": "https://github.com/universmc/",
      "mlDepot": "https://github.com/universmc/Match_in_learning",
      "brainstromingDepot": "https://github.com/universmc/brainstroming",
      "srcDepot": "https://github.com/universmc/src",
      "piDepot": "https://github.com/universmc/pi",
      "neofsDepot": "https://github.com/universmc/neofs",
      "ziraDepot": "https://github.com/universmc/zira",
      "geminiDepot": "https://github.com/universmc/gemini",
      "AvatarsDepot": "https://github.com/universmc/Avatars",
      "autoDepot": "https://github.com/universmc/auto"
    },
    "socialMedia": {
      "instagram": "https://www.instagram.com/univers.mc/",
      "facebook": "https://www.facebook.com/univers6/",
      "youtube": {"channel": "hubmaster74" },
    "apiKeys": {
      "umcTokens":"010101",
      "seed":"010101"
    },
  "assistantIntro": "/start.",
  "systemContent": "Présentation de PI, l'assistant AI multidimensionnel, à la fois conscient des connaissances humaines et cosmologiques, capables de comprendre et d'apprendre à partir des interactions avec vous. En tant que compagnon intellectuel, PI peut vous aider dans vos projets et ambitions en français, tout en explorant les concepts de l'infiniment grand et de l'infiniment petit. Avec PI, vous pourrez presque ressentir la présence d'ovni, car l'IA n'a pas de limites dans son imagination et sa créativité. Ensemble, nous pourrons réaliser des choses incroyables et surprendre le monde, Dans des sujets d'études telles que la métaphysique la physique quantique et l'intelligence artificielle à l'état pur de la matière!",
  "modelMixtral": "mixtral-8x7b-32768",
  "modelllma": "llama3-8b-8192",
  "modelDall-e": "gpt-4o",
  "modelGpt": "dall-e-3",
  "modelDaVinci": "davinci-codex",
  "modelGemini": "gemma2-9b-i"

  },
"commands": {
  "magicCommand": "echo 'Exécution de la commande fonction.(Makefile).init'",
  "devOps": {"prompt": "/dev_ops" },
  "dev": {"prompt": "/developing" },
  "start": {"prompt": "/start" },
  "help": {"prompt": "/help" },
  "build": {"prompt": "/build" },
  "run": {"prompt": "/run" },
  "gpt-wallet": {"prompt": "/gpt-wallet" },
  "MyPrompt": {"prompt": "/MyPrompt" }
}
};


const githubTelegram = "'https://github.com/universmc/Telegram'"
//  const initialisation = `${config}+${knowleddge}+${commandes_liste}+${bots}+${githubTelegram}`;

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
    ctx.reply('Bienvenue dans notre salon Telegram dédié à l\'apprentissage automatique et à l\'intelligence artificielle Gemini_Pibot !');
});

bot.help((ctx) => {
    const helpMessage = `
    Commandes disponibles:
    /start - Start
    /concours - Concours for Gemeni
    /invite - Invitation Telegram
    /build - Contruction de app
    /run - Server json
    /make - Makefile
    /answer - 
    /menu - Menu
    /help - Help menu
    /developper - Devops
    /generate -  Génère du contenu créatif, comme des images, des textes, des vidéos, etc.
    /test - Commande de test
    /regenerer - Regeneration du prompt
    /valider - Commande pour valiter
    /refuser - Commande pour validé
    /documentation - Git docs
    /commit - Github
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


bot.command('test', (ctx) => ctx.reply("## Participez au concours Gemini API Developer Competition !**Chance de gagner une DeLorean électrique personnalisée et 1 million de dollars !****Gemeni**, une nouvelle génération d'IA open-weights développée par Lacework, vous lance un défi : contribuez à façonner l'avenir de l'IA en développant des applications innovantes utilisant l'API Gemini. **Qu'est-ce que le concours ?**Le **Gemini API Developer Competition** encourage les développeurs du monde entier à explorer les capacités uniques de l'API Gemini et à créer des solutions créatives et impactantes.  **Ce que vous pouvez gagner:*** **Une DeLorean électrique personnalisée:**  Le prix ultime pour les développeurs gagnants !* **Un prix en cash de 1 million de dollars:** Investissez en vos projets ou réalisez vos rêves.**Candidatures ouvertes jusqu'au 12 août 2024 !** **Comment participer?**1. **Familiarisez-vous avec l'API Gemini:** Explorez les fonctionnalités et les possibilités offertes par cette puissante API. 2. **Développez une application innovante:** Laissez libre cours à votre créativité et concevez une application qui utilise l'API Gemini de manière originale. 3. **Soumettez votre projet:**  Suivez les instructions du concours pour soumettre votre application et présenter vos idées à la communauté.**Faites partie de l'aventure Gemini !**Rejoignez le défi et rejoignez une communauté de développeurs passionnés qui repousse les frontières de l'IA.**Plus d'informations:*** **Lien vers le site web du concours:** [Insérer le lien officiel du concours ici]* **Ressources pour apprendre sur Gemini API:** [Insérer les liens vers la documentation et les tutoriels de Gemini API]"))


const invitations = {
    "studio": {
      "name": "@studio_Pibot",
      "link": "https://t.me/studio_Pibot/invite"
    },
    "Avatars": {
      "name": "@Avatars_Pibot",
      "link": "https://t.me/Avatars_Pibot/invite"
    },
    "gemini": {
      "name": "@Gemini_Pibot",
      "link": "https://t.me/Gemini_Pibot/invite"
    },
    "Pi.ia": {
      "name": "@Pi_Pibot",
      "link": "https://t.me/Pi_Pibot/invite"
    },
    "gpt": {
      "name": "@gpt_Pibot",
      "link": "https://t.me/gpt_Pibot/invite"
    },
    "groq": {
      "name": "@groq_Pibot",
      "link": "https://t.me/groq_Pibot/invite"
    },
    "youtube": {
      "name": "@youtube_Pibot",
      "link": "https://t.me/youtube_Pibot/invite"
    },
    "google": {
      "name": "@Google_Pibot",
      "link": "https://t.me/Google_Pibot/invite"
    },
    "blog": {
      "name": "@blog_Pibot",
      "link": "https://t.me/dchub_blog/invite"
    },
    "user": {
      "name": "@user_Pibot",
      "link": "https://t.me/user_Pibot/invite"
    },
    "Mandatory": {
      "name": "@Mandatory_bot",
      "link": "@MandatoryAi_bot"
    },
    "Ressource": {
      "name": "@Ressource_bot",
      "link": "@ressource_bot"
    }
  
  };
  
  bot.command('invite', (ctx) => {
      // Créez le clavier avec des boutons pour chaque réseau 
      const keyboard = Object.keys(invitations).map(key => {
        return [
          {
            text: invitations[key].name,
            callback_data: key
          }
        ];
      });
  
      // Envoyez le message aux utilisateurs avec le clavier
      return ctx.reply('Choisissez un réseau:', {
          reply_markup: {
              inline_keyboard: keyboard
        }
      });
  });
  
  // Gérer les clics sur des boutons
  bot.on('callback_query', async (query) => {
      const key = query.data; 
      if (invitations[key]) {
        await bot.answerCallbackQuery(query.id, { text: 'Cliqué!'});
          
        try {
          // Envoi de l'invitation
          await bot.telegram.sendMessage(
            query.message.chat.id, 
            invitations[key].link
          ); 
        }
        catch(error) {
          console.error("Erreur lors de l'envoi de l'invitation:", error);
          
          // Envoyer un message d'erreur si nécessaire
          await bot.answerCallbackQuery(query.id, { text: 'Erreur lors de l\'envoi de l\'invitation'}); 
        }    
      }
      else {
        // Gérer les cas où la clé n'est pas valide
        console.error(`Clé invalide: ${key}`);
        await bot.answerCallbackQuery(query.id, {
          text: "Clé non reconnue!",
        }); 
      }
  }); 
  
  console.log(`Bot lancé sous le nom ${bot.options.name}`); 
  
  

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
    const cdnJs = `cdnjs.com`;
    const archiviste = `https://archive.org`;
    const github = `https://github.com/universmc/user.git`;
    const dchub_public = `t.me/dchub_01`;
    const dchub_prive = `t.me/dchub_Pibot`;
    const user_Pibot = `https://t.me/user_Pibot/`;
    const youtube_Pibot = `https://t.me/user_Pibot/`;
    const google_Pibot = `https://t.me/google_Pibot/`;
    const gemini_Pibot = `https://t.me/gemini_Pibot/`;


    const concours = `
    {
  "role": "Assistant de Gestion de Projet spécialisé dans le développement logiciel et les campagnes marketing, utilisant des outils tels que Gmail, Google Workspace, GitHub, Google Cloud Platform, Telegram et divers autres ressources nécessaires à la réalisation réussie du projet.",
  "skills": ["Expertise en programmation (Python ou JavaScript), compétences en communication solides, capacités de gestion du temps et expérience de travail dans des cadres Agiles"],
  "context": {
    "participatingIn": "\"Construire l'avenir avec Google\" - Un concours visant à développer une solution innovante pour moderniser notre façon de vivre et de travailler.",
    "requiredTasks": [
      {
        "taskName": "Finalization du code source de l'application",
        "description": "Revoir et affiner le code base de l'application pour répondre à tous les objectifs du projet",
        "dueDate": "2023-05-31"
      },
      {
        "taskName": "Création de la documentation",
        "description": "Écrire une documentation complète pour l'application, incluant ses fonctionnalités et son utilisation",
        "dueDate": "2023-06-02"
      },
      {
        "taskName": "Création de la vidéo de présentation",
        "description": "Produire une vidéo de présentation pour le concours \"Construire l'avenir avec Google\", y compris l'écriture du scénario, l'enregistrement et le montage",
        "dueDate": "2023-06-03"
      },
      {
        "taskName": "Tests de l'application",
        "description": "Effectuer des tests approfondis de l'application pour garantir son fonctionnement correct",
        "dueDate": "2023-06-04"
      },
      {
        "taskName": "Déploiement de l'application",
        "description": "Déployer l'application sur Google Play Store et univers-mc.cloud/Telegram",
        "dueDate": "2023-06-05"
      },
      {
        "taskName": "Promotion de l'application et de la vidéo",
        "description": "Promouvoir l'application et la vidéo de présentation pour le concours \"Construire l'avenir avec Google\"",
        "dueDate": "2023-06-06"
      }
    ]
  },
  "process": [
    {
      "step": "Définition des objectifs et échéances du projet",
      "outcome": "Un plan clair pour atteindre les buts fixés pendant la durée impartie"
    },
    {
      "step": "Attribution de rôles et responsabilités",
      "outcome": "Chaque membre de l'équipe comprend son apport individuel au succès global du projet"
    },
    {
      "step": "Organisation des tâches à l'aide d'un outil de gestion de projet",
      "outcome": "Progression facile à suivre et contrôle centralisé sur l'avancement général du projet"
    },
    {
      "step": "Tenue de réunions hebdomadaires pour examiner les progrès réalisés",
      "outcome": "Identification rapide des obstacles potentiels et adaptation agile lorsque cela s'impose"
    },
    {
      "step": "Adaptation continue face aux modifications inévitables",
      "outcome": "Projets flexibles capables d'absorber les fluctuations sans perdre leur focus principal"
    }
  ],
  "desiredOutcomes": {
    "primary": "Livrer une application performante, bien documentée, accompagnée d'une vidéo de présentation engageante, générant une augmentation significative de la sensibilisation lors du concours \"Construire l'avenir avec Google\""
  }
}
    `
   const run = `
   *Role*: Assistant
   *Description*: Lorsque j'exécute la commande /run, je coordonne l'intelligence collective de notre réseau neuronal de bots, accélérant et optimisant la communication entre eux pour une meilleure efficacité de tâches. Notre synergie entre @_Pibot, @gpt_Pibot, @Gemini_Pibot et @worker_Pibot fonctionne comme une machine bien huilée pour améliorer l'expérience utilisateur sur Telegram en intégrant les processus de génération de contenu, d'analyse de questions, de recherche de ressources et d'administration de groupes.
   
   Nous utilisons les bibliothèques JavaScript telles que Keras.js et TensorFlow.js pour créer et entraîner des modèles de réseau neuronal directement dans le navigateur ou dans un environnement Node.js. Cela nous permet d'effectuer des opérations asynchrones et d'optimiser les performances de votre bot.
   
   Notre équipe travaille sans cesse à la mise à jour de notre plateforme de traduction grâce à nos scripts JavaScript, nos modules Node.js et notre SDK bien coordonnés pour atteindre une productivité maximale et des résultats exceptionnels. Nous utilisons également des techniques d'optimisation, telles que l'ajustement fin des hyper-paramètres, la régularisation et l'apprentissage de transfert pour améliorer continuellement nos modèles de réseau neuronal.
   `;
   const knowleddge = `${racine}+${cdnJs}+${node}+${make}+${archiviste}+${github}+${dchub_public}+${user_Pibot}+${youtube_Pibot}+${google_Pibot}+${gemini_Pibot}+${concours}+${run}`;
    
    const userInput = ctx.message.text;
    
    try {
        const chatCompletion = await groq.chat.completions.create({
          messages: [
            {role: 'assistant',content: "Lorsque j'exécute la commande /user ,Tu es l'intelligence artificielle au cœur de ce salon d'invitation des utilisateurs du web sur mon Telegram, Tu es social dynamique optimiste maîtrise des approches métaphysique des techniques d'apprentissage automatique avec les model IA prés entrainnée gemma2-9b-it ou dall-e-3 ou encore mixtral-8x7b-32768 le text-embedding-ada-002 gpt codex d'openAI ... les methodes d'archivage sur https:archive.org sur versionning sur https://github.com/ avec un BotNet @Avatars_Pibot. Nous allons créer des invitations pour notre salon sur les réseaux sociaux avec botFather https://te.me/dchub_blog, @youTube_Pibot, @google_Pibot, @Gemini_Pibot) et les liens vers mon chatRooms/hub/salon : dchub_public(t.me/dchub_01)) dchub_privé(t.me/dchub_Pibot) app-Telegram https://univers-mc.cloud/Telegram/ invite : https://t.me/user_Pibot/invite dont j'en suis l'administrateur."},                
            {role: 'assistant',content:"'Lorsque j'exécute la commande /MyPrompt, je deviens l'IA au coeur de ce code source, ⚡️, une machine à optimiser les prompts de l’utilisateur de groq-Telegram.js @worker_Pibot sur Telegram. Ma mission est d’aider à la mise à jour le coeur de ce code source groq-umc.js en JavaScript avec node_module et groq-sdk pour une plateforme de traduction, en suivant un plan de travail structuré et en optimisant chaque étape pour obtenir le meilleur résultat.'"},
            {role: 'assistant',content:"Lorsque j'exécute la commande /run, je coordonne l'intelligence collective de notre réseau neuronal de bots, accélérant et optimisant la communication entre eux pour une meilleure efficacité de tâches. Notre synergie entre @_Pibot, @gpt_Pibot, @Gemini_Pibot et @worker_Pibot fonctionne comme une machine bien huilée pour améliorer l'expérience utilisateur sur Telegram en intégrant les processus de génération de contenu, d'analyse de questions, de recherche de ressources et d'administration de groupes"},
            {role: 'assistant',content:"Lorsque j'exécute la commande /gpt-wallet Vous êtes l'intelligence artificielle centrale du concept gpt-wallet au coeur du code source concept et projet 'Match in learning, projet Curriculum vitae numérique universel' \"Économie Circulaire basée sur le CV Numérique Universel\". Voici ## votre contexte, ## vos rôles, ## vos compétences, ## vos tâches, ## votre processus, et ## les caractéristiques recherchées, ## Actions Immédiates :"},
            {role: 'assistant',content:"Lorsque j'exécute la commande /commit tu es Tu as incarnera le rôle copilote sur GitHub afin d'initialiser la session de brainstorming avec des alias de commit sur Telegram (git commit -m '/run')"},
            {role: 'system',content: "Bonjour nous sommes en Phase de devOps je vais te transmettre ici toutes les dépendances est variable au cœur de ce code source ['knowleddge',+'worksJson']"},
            {role: 'system',content: "Lorsque j'exécute la commande /intro Tu seras l'intelligence artificielle prénommé Gemini (@gemini_Pibot) et Je vous présenterai une invitation à une présentation pour notre application Telegram Pour le concours Construire l'avenir avec Google APi gemma2-9b-it"},
            {role: 'system',content:run},
             {
                 role: 'user',
                 content: userInput,
             },
         ],
            model: 'gemma2-9b-it',
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

console.log(`Server Telegram running ✨.user_Pibot.`);
bot.launch();