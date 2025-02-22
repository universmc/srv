## Guide étape par étape pour créer un modèle d'article WordPress avec CSS personnalisé

Ce guide vous aidera à comprendre et à réaliser un modèle d'article WordPress personnalisé avec du CSS. Il est conçu pour les débutants et les utilisateurs intermédiaires qui souhaitent apprendre à contrôler l'apparence de leurs articles WordPress.

**Prérequis:**

* Connaissance de base de WordPress
* Accès à un site WordPress
* Connaissance de base du langage HTML et CSS (optionnel, mais recommandé)

**Étapes:**

1. **Créer un fichier PHP pour votre modèle d'article:**

   * Connectez-vous à votre site WordPress via l'interface d'administration.
   * Accédez à **Apparence > Thèmes > Modifier** le thème actif.
   * Créez un nouveau fichier PHP dans le dossier `templates` de votre thème. Nommez ce fichier `article.php`.

2. **Ajouter le code HTML de base à votre fichier `article.php`:**

   * Ouvrez le fichier `article.php` dans un éditeur de texte.
   * Collez le code HTML suivant à l'intérieur du fichier :

 ```php
 <?php 
  while ( have_posts() ) : the_post(); 
?>
  <h1><?php the_title(); ?></h1>
  <p><?php the_content(); ?></p>
<?php 
  endwhile;
?>
 ```

   * Ce code affiche le titre et le contenu de l'article actuel.

3. **Ajouter du CSS personnalisé:**

   * Vous pouvez ajouter du CSS directement dans votre fichier `article.php` en utilisant des balises `<style>`.
   * Par exemple, pour changer la couleur du titre de l'article en rouge, vous pouvez ajouter le code suivant :

 ```php
 <style>
   h1 {
     color: red;
   }
 </style>
 ```

4. **Créer un fichier CSS séparé:**

   * Il est recommandé de créer un fichier CSS séparé pour votre modèle d'article afin de le maintenir propre et organisé.

   * Créez un nouveau fichier CSS dans le dossier `css` de votre thème. Nommez ce fichier `article.css`.

5. **Ajouter des règles CSS à votre fichier `article.css`:**

   * Ouvrez le fichier `article.css` dans un éditeur de texte.
   * Ajoutez les règles CSS que vous souhaitez appliquer à votre modèle d'article.

   * Par exemple, vous pouvez ajouter des règles pour changer la police, la taille de police, la couleur de fond, etc.

6. **Inclure votre fichier CSS dans votre fichier `article.php`:**

   * Ajoutez la ligne de code suivante à votre fichier `article.php` pour inclure votre fichier CSS :

 ```php
 <link rel="stylesheet" href="<?php bloginfo('stylesheet_directory'); ?>/css/article.css">
 ```

**Conseils:**

* Utilisez des commentaires dans votre code pour expliquer ce que chaque partie fait.
* Testez votre modèle d'article sur différents navigateurs pour vous assurer qu'il s'affiche correctement.
* N'hésitez pas à expérimenter avec différents styles et couleurs pour trouver l'apparence que vous souhaitez.

**Ressources supplémentaires:**

* [Documentation WordPress sur les modèles](https://developer.wordpress.org/themes/basics/template-files/)
* [Tutoriels WordPress sur le CSS](https://www.wpbeginner.com/wp-themes/how-to-add-custom-css-to-wordpress/)



