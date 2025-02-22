# galileo
Internet, par satellite


## Guide d'installation - Application Galileo Satellite Tracker
Ce guide vous aidera à configurer et à démarrer votre application Electron-Node.js pour le suivi en temps réel des satellites Galileo. Suivez les étapes ci-dessous pour installer toutes les dépendances nécessaires et exécuter votre application.

## Prérequis
Avant de commencer, assurez-vous d'avoir les éléments suivants installés sur votre machine :


Node.js (version >= 8) : Téléchargez et installez depuis nodejs.org.
npm (inclus avec Node.js) ou yarn : Pour gérer les packages.
Electron : Pour exécuter l'interface de bureau.
Python (si vous travaillez avec PyTorch ou TensorFlow pour le machine learning).
Git : Pour cloner et gérer le code source.
Vérification des installations :
Ouvrez un terminal et exécutez les commandes suivantes pour vérifier que tout est bien installé :

<pre>
<code>node -v</code>
<code>npm -v<c/ode>
<code>git --version</code>
</pre>

# Étape 1 : Cloner le projet
Clonez le projet depuis votre dépôt GitHub (ou un autre emplacement).

<pre>
<code>git clone <https://github.com/universmc/galileo.git></code>
<code>cd galileo</code>
</pre>

# Étape 2 : Installation des dépendances 
Utilisez npm ou yarn pour installer les dépendances listées dans le fichier package.json.

Installation avec npm :
<pre>
<code>npm install</code>
</pre>
>
Ou installation avec yarn :

<pre>
<code>yarn install</code>
</pre>

Cette étape installera toutes les dépendances nécessaires, y compris celles pour Electron, Next.js, React, et d'autres outils pour la visualisation et le traitement des données.

# Étape 3 : Démarrage de l'application
En mode développement :
Pour lancer l'application en mode développement, utilisez la commande suivante :

<pre>
<code>npm run dev</code>
</pre>

Cette commande démarrera un serveur local pour l'application avec Next.js, qui est utilisé pour la partie frontend.

Exécuter Electron (interface utilisateur desktop) :
Pour lancer l'application via Electron, utilisez la commande suivante :

<pre>
<code>npm start</code>
</pre>

Compilation de CSS avec Sass :
Si vous devez observer et compiler des fichiers SCSS en CSS :

bash
Copier le code
npm run style
Construction de l'application pour la production :
Pour construire l'application en vue d'un déploiement en production :

bash
Copier le code
npm run build
Cette commande compile l'application pour une distribution.

Étape 4 : Scripts supplémentaires
Linting : Pour vérifier et corriger les erreurs de code avec ESLint, utilisez :

bash
Copier le code
npm run lint
Script map : Utilisez ce script pour exécuter le programme de cartographie avec Node.js :

bash
Copier le code
npm run map
Étape 5 : Modèles d'IA (Optionnel)
L'application utilise différents modèles d'intelligence artificielle configurés dans package.json, tels que Mistral, Llama, Gemma et GPT. Vous pouvez ajuster les paramètres de ces modèles (comme temperature, max_tokens, etc.) en fonction de vos besoins.

Les modèles peuvent être testés avec les scripts suivants (exemple pour le modèle Gemma) :

bash
Copier le code
npm run gemma
Ces modèles nécessitent l'installation de PyTorch ou d'autres frameworks, que vous pouvez gérer séparément selon votre configuration de machine learning.

Étape 6 : Dépendances supplémentaires
Voici quelques-unes des dépendances clés utilisées dans le projet :

@mapbox/vector-tile : Pour manipuler les tuiles vectorielles.
Leaflet.js ou Cesium.js : Pour la visualisation des cartes (assurez-vous de bien intégrer ces bibliothèques si besoin).
Next.js : Pour la partie serveur et frontend.
Electron : Pour la partie application de bureau.
OpenAI : Pour interagir avec les modèles GPT.
Dépendances spécifiques :
Voici un aperçu rapide de certaines dépendances clés installées par ce projet :

Axios : Pour gérer les requêtes HTTP vers des API externes.
Jimp : Pour manipuler les images dans Node.js.
Telegraf : Pour intégrer des bots Telegram.
Tone.js : Pour la gestion de l'audio.
Pytorch : Pour le machine learning avec les modèles d'IA.
Étape 7 : Déploiement sur Vercel
Si vous souhaitez déployer le frontend sur Vercel (service de déploiement de Next.js), vous pouvez le faire en utilisant :

------------------
vercel
Cela créera un projet déployé sur Vercel, prêt à être consulté en ligne.

