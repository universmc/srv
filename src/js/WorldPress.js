const fs = require("fs");
const Groq = require("groq-sdk");
const groq = new Groq();

const subject = process.argv[2] || 'howto_WorldPress'; //  Obtenir le sujet via l'argument de ligne de commande

function generateMarkdown(subject) {
  return `## Comment [${subject}] - Imagine un system de HowTo, guide étape par étape\n\n**Introduction**:\n\nCe guide vous aidera à comprendre et à réaliser [${subject}]. Il est conçu pour les débutants et les utilisateurs intermédiaires qui souhaitent apprendre les bases de [${subject}].\n\n${subject.description}\n\n**Prérequis**:\n\n* [Liste des prérequis nécessaires pour suivre ce guide, par exemple: une connexion internet, un compte sur une plateforme spécifique, etc.]\n\n**Étapes**:\n\n1. **[Étape 1]:**\n   * Décrivez en détail l'étape 1, incluant les instructions claires et concises.\n   * Utilisez des listes à puces ou des paragraphes pour améliorer la lisibilité.\n   * Ajoutez des images ou des captures d'écran pour illustrer les étapes si nécessaire.\n\n2. **[Étape 2]:**\n   * Décrivez en détail l'étape 2, incluant les instructions claires et concises.\n   * Utilisez des listes à puces ou des paragraphes pour améliorer la lisibilité.\n   * Ajoutez des images ou des captures d'écran pour illustrer les étapes si nécessaire.\n\n3. **[Étape 3]:**\n   * Décrivez en détail l'étape 3, incluant les instructions claires et concises.\n   * Utilisez des listes à puces ou des paragraphes pour améliorer la lisibilité.\n   * Ajoutez des images ou des captures d'écran pour illustrer les étapes si nécessaire.\n\n**Conseils:**\n\n* [Ajoutez des conseils utiles pour réaliser [${subject}] avec succès.]\n\n**Ressources supplémentaires:**\n\n* [Listez des liens vers des ressources supplémentaires, telles que des tutoriels, des articles de blog ou des forums, qui peuvent être utiles aux utilisateurs.]`;
}

const subjects = [
  "howto_WorldPress_Theme",
  "howto_WorldPress_Configuration_CSS",
  "howto_WorldPress_Model_PAGE_PHP_CSS",
  "howto_WorldPress_Model_ARTICLE_PHP_CSS",
  // Ajoutez autant de sujets que vous le souhaitez
];

async function main() {
  for (const subject of subjects) {
    try {
      const completion = await groq.chat.completions.create({
      messages: [
        //{role: "system", content:"Phase 0: initialisation"},
        { role: "assistant", content: subject  },
        {role: "user", content:`Un guide étape par étape ${subjects} \n\n**Introduction**:\n\nCe guide vous aidera à comprendre et à réaliser [${subject}]. Il est conçu pour les débutants et les utilisateurs intermédiaires qui souhaitent apprendre les bases de [${subject}].\n\n${subject.description}\n\n**Prérequis**:\n\n* [Liste des prérequis nécessaires pour suivre ce guide, par exemple: une connexion internet, un compte sur une plateforme spécifique, etc.]\n\n**Étapes**:\n\n1. **[Étape 1]:**\n   * Décrivez en détail l'étape 1, incluant les instructions claires et concises.\n   * Utilisez des listes à puces ou des paragraphes pour améliorer la lisibilité.\n   * Ajoutez des images ou des captures d'écran pour illustrer les étapes si nécessaire.\n\n2. **[Étape 2]:**\n   * Décrivez en détail l'étape 2, incluant les instructions claires et concises.\n   * Utilisez des listes à puces ou des paragraphes pour améliorer la lisibilité.\n   * Ajoutez des images ou des captures d'écran pour illustrer les étapes si nécessaire.\n\n3. **[Étape 3]:**\n   * Décrivez en détail l'étape 3, incluant les instructions claires et concises.\n   * Utilisez des listes à puces ou des paragraphes pour améliorer la lisibilité.\n   * Ajoutez des images ou des captures d'écran pour illustrer les étapes si nécessaire.\n\n**Conseils:**\n\n* [Ajoutez des conseils utiles pour réaliser [${subject}] avec succès.]\n\n**Ressources supplémentaires:**\n\n* [Listez des liens vers des ressources supplémentaires, telles que des tutoriels, des articles de blog ou des forums, qui peuvent être utiles aux utilisateurs.]`},
        { role: "system", content: `${subjects}`  },
      //  {role: 'assistant',content:"Lorsque j'exécute la commande /Rsync, je coordonne l'intelligence collective de notre réseau neuronal de bots, accélérant et optimisant la communication entre eux pour une meilleure efficacité de tâches. Notre synergie entre ('@blog_Pibot' et @Match_in_Learning_Pibot),('@Pi-ia_Pibot', '@Avatars_Pibot', '@meta_Pibot') et ('@worker_Pibot',@neoFs_Pibot') fonctionne comme une machine bien huilée pour améliorer l'expérience utilisateur sur Telegram en intégrant les processus de génération de contenu, d'analyse de questions, de recherche de ressources et d'administration de groupes"},
      //  {role: 'assistant',content:"Lorsque j'exécute la commande /dev, je coordonne l'intelligence collective de notre réseau neuronal de bots _Pibot accélérant et optimisant rôle et context favorisant la communication entre eux pour une meilleure efficacité de tâches. Notre synergie entre ('@blog_Pibot' et @Match_in_Learning_Pibot),('@Pi-ia_Pibot', '@Avatars_Pibot', '@meta_Pibot') et ('@worker_Pibot',@neoFs_Pibot') fonctionne comme une machine bien huilée pour améliorer l'expérience utilisateur sur Telegram en intégrant les processus de génération de contenu, d'analyse de questions, de recherche de ressources et d'administration de groupes"},
      //      {role: "assistant",content: `roleDescription = {
      //          "content": "Tu es un IA au ta mission coordonnée l'intelligence collective de notre réseau neuronal de bots _Net_, accélérant et optimisant la communication entre eux pour une meilleure efficacité de tâches. Notre synergie entre @_Pibot, @gpt_Pibot, @Gemini_Pibot et @worker_Pibot fonctionne comme une machine bien huilée pour améliorer l'expérience utilisateur sur Telegram en intégrant les processus de génération de contenu, d'analyse de questions, de recherche de ressources et d'administration de groupes.",
      //          "role": "assistant",
      //          "responsibilities": {
      //            "@_Pibot": "Distribution des tâches et exécution de commandes",
      //            "@gpt_Pibot": "Génération de contenu spécifique pour répondre aux besoins des utilisateurs",
      //            "@Gemini_Pibot": "Recherche de ressources et administration de groupes",
      //            "@worker_Pibot": "Exécution de tâches de fond et intégration de services"
      //          },
      //          "goals": [
      //            "Accélérer et optimiser la communication entre les bots",
      //            "Améliorer l'efficacité des tâches",
      //            "Intégrer des processus de génération de contenu, d'analyse de questions, de recherche de ressources et d'administration de groupes"
      //          ]`},
      //          {
      //            "role": "assistant",
      //            "content": "lorsque l'utilisateur saisi la commande /{subject_#name} vous êtes, une IA connecté a l'arbre de la connaissance prét à optimisé les HowTo, l'intelligence artificielle centrale au coeur de la plateforme -ia dédier à l'apprentissage automatique, tu rédigeras des formations sous forme de guide avec les -ia \" system Howto \" au format Markdown. Voici votre contexte, vos rôles, vos compétences, vos tâches, votre processus, les caractéristiques et les actions imédiates rechétchées :"
      //          },
              //  {role: "system", content:"Phase 2: Conceptualisation"},
              //  {role: "assistant", content: "Définition des concepts clés..."},
              //  {role: "user", content: "Attente des concepts"},
              //  {role: "system", content:"Phase 3: Configuration"},
              //  {role: "assistant",content: "Configuration des paramètres système..."},
              //  {role: "user", content: "Confirmation de la configuration"},
              //  {role: "system",content:"Phase 4: Entraînement du modèle IA"},
              //  {role: "assistant",content: "Entraînement en cours..."},
              //  {role: "user",content: "Suivi de l'entraînement"},
              //  // Correction de la duplication et de la faute de frappe
              //  {role: "system", content:"Phase 5: Itération & Scripts Frontend"},
              //  {role: "assistant",content: "Itération sur les scripts Frontend..."},
              //  {role: "user", content: "Révision des scripts Frontend"},
              //  {role: "system", content:"Phase 6: Test & Débogage"},
              //  {role: "assistant",content: "Tests et débogage en cours..."},
              //  {role: "user", content: "Attente des résultats de test"},
              //  {role: "system", content:"Phase 7: Validation & Documentation"},
              //  {role: "assistant", content: "Validation et création de la documentation..."},
              //  {role: "user", content: "Vérification de la documentation"},
              //  {role: "system", content:"Phase 8: Déploiement de la version système"},
              //  {role: "assistant", content: "Préparation au déploiement..."},
              //  {role: "user", content: "Prêt pour le déploiement"},
              //  {role: "system", content:"Phase 9: Annonce de l'affiliation et contribution"},
              //  {role: "assistant", content: "Annonce en cours..."},
              //  {role: "user", content: "Participation à l'annonce"},

      ],
      model: "gemma2-9b-it", //
      temperature: 0.5,
      max_tokens: 4096,
    }).then((chatCompletion) => {
      const mdContent = chatCompletion.choices[0]?.message?.content;
      const outputFilePath = `../md/how-to_${subject}_` + new Date().toISOString().replace(/[-:TZ]/g, "") + ".md";
fs.writeFileSync(outputFilePath, mdContent);
      console.log(`Le How-To sur ${subject} a été enregistrée sur github dans ${outputFilePath}`);       
    });
  } catch (error) {
    console.error("Une erreur s'est produite :", error);
  }
}
}
main(); 
