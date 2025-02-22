const { exec } = require('child_process');
const fs = require('fs');

// Fonction pour générer la suite de Fibonacci
function fibonacci(n) {
  let fib = [0, 1];
  for (let i = 2; i <= n; i++) {
    fib[i] = fib[i - 1] + fib[i - 2];
  }
  return fib;
}

// Fonction pour compiler le fichier SCSS
async function compileSCSS() {
  return new Promise((resolve, reject) => {
    exec('npm run compileSCSS', (error, stdout, stderr) => {
      if (error) {
        console.error(`Erreur de compilation SCSS: ${error.message}`);
        return reject(error);
      }
      if (stderr) {
        console.error(`Erreur de sortie SCSS: ${stderr}`);
      }
      console.log(`Compilation SCSS réussie : ${stdout}`);
      resolve();
    });
  });
}

// Fonction pour gérer l'exécution asynchrone des étapes du plan de travail
async function executeTask(task, taskIndex, workPlan) {
  console.log(`Démarrage de la tâche: ${task.sequence}`);
  console.log(`Description: ${task.description}`);

  const completedSteps = [];

  for (const step of task.steps) {
    if (!completedSteps.includes(step)) {
      console.log(`Exécution de l'étape: ${step}`);
      await executeStep(step);
      completedSteps.push(step);
    }
  }
  console.log(`✨ }--------- Tâche terminée: ${task.sequence}`);
  console.log(' ✨ -------════════════════════════════-------{🔷}---------═══════════════════════════------- ✨ ');

  saveProgress(taskIndex, workPlan, completedSteps);
}

// Simule l'exécution asynchrone d'une étape
async function executeStep(step) {
  return new Promise((resolve) => {
    setTimeout(() => {
      console.log(`✨ }---- Étape terminée: ${step}`);
      resolve();
    }, 1000);
  });
}

// Fonction pour exécuter toutes les tâches dans le plan de travail
async function executeWorkPlan(workPlan) {
  let cycle = 0;
  const fibSeries = fibonacci(15);  // Génére une série Fibonacci pour moduler la durée du cycle
  let fibIndex = 0;

  while (true) {
    console.log(`Démarrage du cycle : ${cycle + 1}`);
    
    for (let i = 0; i < workPlan.tasks.length; i++) {
      await executeTask(workPlan.tasks[i], i, workPlan);
    }

    console.log(`Cycle ${cycle + 1} terminé.`);

    // Compiler le SCSS après chaque cycle
    await compileSCSS();

    // Incrémenter le compteur de cycle
    cycle++;

    // Sélectionner la durée basée sur Fibonacci pour moduler la pause
    const pauseDuration = fibSeries[fibIndex % fibSeries.length] * 1000;
    fibIndex++;

    console.log(`Pause de ${pauseDuration / 1000} secondes avant de recommencer...`);

    await new Promise(resolve => setTimeout(resolve, pauseDuration));
  }
}

// Fonction pour sauvegarder la progression dans un fichier JSON
function saveProgress(taskIndex, workPlan, completedSteps) {
  const progress = {
    currentTaskIndex: taskIndex,
    completedSteps: completedSteps,
    timestamp: new Date().toISOString(),
    status: workPlan.tasks[taskIndex].status,
    progress: workPlan.tasks[taskIndex].progress
  };

  fs.writeFileSync('progress.json', JSON.stringify(progress, null, 2));
  console.log(`Progression sauvegardée: Tâche ${taskIndex}`);
}

// Structure du plan de travail
const workPlan = 
{
  "tasks": [
    {
      "phase": "Phase 1: Préparation",
      "location": "Open Space",
      "time_start": "2024-10-15T04:00:00",
      "time_end": "2024-10-15T05:00:00",
      "description": "Préparation de l'environnement de travail, synchronisation des fichiers et outils nécessaires.",
      "steps": [
        "Organiser les fichiers de projet (Electron, GPT-Wallet, CVUN, Match in Learning).",
        "Vérifier les dépendances des projets via Makefile.",
        "Configurer et tester les scripts de build dans workflow.js.",
        "Préparer await.json pour la gestion asynchrone des tâches.",
        "Vérifier les scripts de déploiement sur le serveur privé.",
        "Préparer les modules vidéo pour les réseaux sociaux (ShortVideoValue)."
      ],
      "status": "Terminé",
      "progress": "100%"
    },
    {
      "phase": "Phase 2: Travaux sur GPT-Wallet",
      "location": "Open Space",
      "time_start": "2024-10-15T05:00:00",
      "time_end": "2024-10-15T07:00:00",
      "description": "Développement et optimisation du projet GPT-Wallet pour monétiser les compétences via des Smart Contracts.",
      "steps": [
        "Finaliser les Smart Contracts (umcTokens.sol).",
        "Intégrer les transactions Pi Coins pour la monétisation des compétences.",
        "Améliorer les interactions utilisateur avec l'assistant AI (@MandatoryAi_bot).",
        "Rédiger la documentation pour les utilisateurs du GPT-Wallet."
      ],
      "status": "En cours",
      "progress": "10%"
    },
    {
      "phase": "Phase 3: Présentation du projet à l'université",
      "location": "Université",
      "time_start": "2024-10-15T07:00:00",
      "time_end": "2024-10-15T09:00:00",
      "description": "Présentation du projet et collecte de données à l'université.",
      "steps": [
        "Préparer et présenter le questionnaire sur la réforme du Code du Travail.",
        "Discuter avec des citoyens et universitaires pour obtenir des retours.",
        "Recueillir des données sur la valeur travail et le curriculum numérique.",
        "Présenter la solution d'allocation universelle mensuelle basée sur le CVUN."
      ],
      "status": "En cours",
      "progress": "0%"
    },
    {
      "phase": "Phase 4: Pause",
      "location": "Open Space",
      "time_start": "2024-10-15T09:00:00",
      "time_end": "2024-10-15T09:30:00",
      "description": "Pause pour se détendre et réinitialiser l'esprit.",
      "steps": [
        "Bilan de la phase précédente.",
        "Analyse des travaux réalisés.",
        "Planification des prochaines étapes."
      ],
      "status": "Terminé",
      "progress": "100%"
    },
    {
      "phase": "Phase 5: Récolte d'idées à l'Open Space",
      "location": "Open Space",
      "time_start": "2024-10-15T09:30:00",
      "time_end": "2024-10-15T11:30:00",
      "description": "Récolte d'idées et ressources pour le projet.",
      "steps": [
        "Discussion avec des professionnels et utilisateurs sur les méthodes de développement.",
        "Recueillir des suggestions pour améliorer le projet.",
        "Développer des méthodes et données pour l'optimisation du projet."
      ],
      "status": "En cours",
      "progress": "50%"
    },
    {
      "phase": "Phase 6: Tribunal pour l'aide juridictionnelle",
      "location": "Tribunal",
      "time_start": "2024-10-15T11:30:00",
      "time_end": "2024-10-15T12:30:00",
      "description": "Se rendre au tribunal civil pour obtenir l'aide juridictionnelle.",
      "steps": [
        "Obtenir des informations sur les procédures pour obtenir l'aide juridictionnelle.",
        "Rédiger les courriers nécessaires pour présenter le cas.",
        "Discuter de la défense face à la répression financière."
      ],
      "status": "En cours",
      "progress": "40%"
    },
    {
      "phase": "Phase 7: Documentation et finalisation",
      "location": "Open Space",
      "time_start": "2024-10-15T12:30:00",
      "time_end": "2024-10-15T13:30:00",
      "description": "Finalisation de la documentation technique et utilisateur pour les projets.",
      "steps": [
        "Rédiger la documentation utilisateur pour GPT-Wallet et CVUN.",
        "Documenter les parcours d'apprentissage dans Match in Learning.",
        "Finaliser la documentation sur l'intégration des assistants IA dans les projets."
      ],
      "status": "En cours",
      "progress": "60%"
    },
    {
      "phase": "Phase 8: Récapitulatif et planification des étapes suivantes",
      "location": "Open Space",
      "time_start": "2024-10-15T13:30:00",
      "time_end": "2024-10-15T14:00:00",
      "description": "Bilan des tâches accomplies et planification des prochaines étapes.",
      "steps": [
        "Revoir les objectifs atteints pendant la journée.",
        "Planifier les étapes suivantes pour chaque projet (CVUN, GPT-Wallet, Match in Learning).",
        "Préparer le calendrier pour les prochains développements."
      ],
      "status": "En cours",
      "progress": "10%"
    }
  ]
};
  

// Exécution infinie du plan de travail avec pause Fibonacci
executeWorkPlan(workPlan);
