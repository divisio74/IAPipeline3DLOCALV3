# Un jeu vidéo généré par l'IA, joueur par joueur

> Document de vision, rédigé le 4 octobre 2026 à partir des explications de l'auteur du projet.
> Il décrit **le but final** et **comment l'ensemble pourrait fonctionner**. L'état d'avancement réel
> est résumé à la fin (section « Où on en est »).

---

## 1. L'idée en une phrase

Un jeu vidéo dont le contenu (monstres, objets, attaques, animations, sons **et le code qui les fait
fonctionner dans le jeu**) est **inventé et fabriqué par des IA au moment où le joueur en a besoin**,
utilisé le temps d'un donjon, puis **jeté**, parce qu'on peut en refaire à l'infini, vite et pour peu
de calcul.

## 2. Ce qui change par rapport à un jeu classique

| Jeu classique | Ce jeu |
|---|---|
| Des artistes modélisent un bestiaire fini (50, 200, 1 000 monstres) | Le bestiaire est **infini** : chaque monstre est **unique**, créé pour **un seul joueur** |
| Le contenu est stocké et réutilisé pendant des années | Le contenu est **jetable** : il existe le temps d'un donjon, puis il est supprimé |
| Les comportements sont codés à la main par des développeurs | Chaque monstre ou objet arrive **avec son propre code de comportement**, écrit par un modèle de langage |
| Ajouter du contenu coûte du temps humain | Ajouter du contenu coûte **quelques minutes de calcul** |

Le joueur ne combat jamais deux fois la même créature. Personne d'autre ne la verra jamais.
C'est ce qui rend l'expérience vraiment « libre » : le jeu n'est pas limité par ce que l'équipe a eu
le temps de fabriquer.

## 3. Le principe : des « usines » à contenu

Le jeu repose sur **plusieurs pipelines**, une par type de contenu. Chacune est une usine
automatique : elle reçoit une demande (« un monstre pour le donjon de glace, niveau 12 ») et livre
un **paquet prêt à brancher** dans le moteur du jeu.

| Pipeline | Ce qu'elle produit | Difficulté |
|---|---|---|
| **Monstres** (en cours) | modèle 3D texturé, animations (marche, attaques…), sons, code de comportement | la plus complexe |
| **Objets / items** (prévue) | modèle 3D, éventuelle animation (arme qui s'allume, coffre qui s'ouvre), sons, code d'interaction | beaucoup plus légère |
| D'autres plus tard | décors de donjon, pièges, PNJ, quêtes… | à définir |

On commence par la pipeline monstres **parce que c'est la plus dure** : un monstre a une anatomie,
doit bouger de façon crédible et doit attaquer. Tout ce qu'on y apprend (génération 3D, contrôle
qualité automatique, génération de code) servira ensuite aux autres pipelines, plus simples.

## 4. Le « paquet » livré au jeu

Chaque pipeline livre la même chose : un **paquet d'asset**, toujours au même format, que le moteur
sait charger sans intervention humaine.

```
monstre_7f3a2c/
├── manifeste.json        # carte d'identité : nom, description, stats, attaques, liens vers tout le reste
├── modele.glb            # maillage + texture + squelette + clips d'animation nommés
├── sons/
│   ├── cri.ogg
│   ├── pas.ogg
│   └── attaque_morsure.ogg
└── comportement.lua      # (ou autre langage de script) : le code généré, propre à CE monstre
```

Le **manifeste** est le contrat entre l'IA et le jeu. Par exemple :

```json
{
  "id": "monstre_7f3a2c",
  "nom": "Rôdeur des cendres",
  "description": "Loup démoniaque à la fourrure de charbon, crête d'os sur l'échine.",
  "niveau": 12,
  "stats": { "pv": 340, "vitesse": 4.2, "armure": 8 },
  "animations": ["idle", "marche", "course", "morsure", "charge", "mort"],
  "attaques": [
    { "nom": "Morsure", "animation": "morsure", "degats": 22, "portee": 1.5,
      "evenement_impact": 0.42, "son": "sons/attaque_morsure.ogg" },
    { "nom": "Charge", "animation": "charge", "degats": 35, "portee": 8,
      "recharge": 6.0 }
  ],
  "comportement": "comportement.lua"
}
```

Le jeu n'a besoin de rien savoir d'autre : il charge le modèle, lit le manifeste et exécute le
script dans un bac à sable (voir §7).

## 5. Comment un monstre est fabriqué (pipeline monstres)

L'organisation suit le principe déjà en place dans le projet :
**le modèle de langage (LLM) est le cerveau, le modèle de vision (VLM) les yeux, le code les mains,
et une fiche partagée la mémoire**. La fiche suit le monstre d'une étape à l'autre, et chaque étape y
lit le contexte et y ajoute ce qu'elle a produit.

```
 demande du jeu ─► 1. CRÉATEUR (LLM) ─► fiche du monstre
                                             │
                    2. CONCEPT (image) ◄─────┤
                    3. VIDÉO DE TOUR ◄───────┤
                    4. MODÈLE 3D + TEXTURE ◄─┤
                    5. DIRECTEUR (VLM) ◄─────┤   rapport : membres, objets, articulations
                    6. ANIMATIONS ◄──────────┤
                    7. SONS ◄────────────────┤
                    8. CODE DE COMPORTEMENT ◄┤
                    9. OPTIMISATION + EXPORT ┘─► paquet d'asset ─► jeu
```

1. **Créateur (LLM)** : il *invente* un monstre original, cohérent et effrayant, sans piocher dans une
   liste figée. Il reçoit les contraintes du jeu (thème du donjon, niveau, budget de puissance) et
   le périmètre de complexité (de l'humanoïde à l'araignée ; Slenderman ≈ le plafond). Il écrit la
   fiche : description, anatomie, matières, ce qui fait peur, **pose de référence** (chaque membre
   séparé du corps par un vide visible, pour pouvoir l'animer), liste d'actions.
2. **Concept** : une image du monstre de face, générée depuis la fiche (aujourd'hui Qwen-Image-2.1).
3. **Vidéo de tour** : une caméra fait le tour du monstre (Wan 2.2 + LoRA « orbite »). C'est ce qui
   donne des vues **cohérentes** de tous les côtés, y compris le dos.
4. **Modèle 3D** : reconstruction depuis 4 vues choisies dans la vidéo (Pixal3D multi-vues), puis
   texture.
5. **Directeur (VLM)** : il regarde le modèle et rédige le **rapport** : quels membres, combien,
   lesquels sont articulés, quels objets sont tenus. Ce rapport ne corrige rien. Il **décrit** le modèle,
   pour que les étapes suivantes s'adaptent au corps réel, quel qu'il soit. Rien n'est codé pour un
   type de créature précis.
6. **Animations** : pour chaque action, le LLM écrit une consigne adaptée au corps (allure de marche,
   façon d'attaquer), une IA vidéo filme l'action depuis le modèle, puis un modèle d'animation
   transfère ce mouvement sur le maillage ou sur un squelette.
7. **Sons** : cris, pas, impacts, générés à partir de la fiche.
8. **Code de comportement** : voir §6.
9. **Optimisation + export** : réduction du nombre de polygones au budget du jeu, texture compressée,
   squelette et clips nommés, assemblage du paquet.

À chaque étape, des **contrôles automatiques** (mesures + VLM) vérifient le résultat. En cas d'échec,
l'étape est relancée un nombre limité de fois, puis on passe à une solution de secours plutôt que de
bloquer le joueur.

## 6. Le code généré : la partie la plus nouvelle

C'est ce qui pousse le concept plus loin que les générateurs d'assets existants : **un monstre n'est
pas seulement un modèle animé, c'est aussi un comportement**. Le LLM de la pipeline écrit le script
qui dit au jeu *comment ce monstre se bat*.

### Ce que le LLM sait (son « prompt système »)

Le LLM ne devine rien. Son prompt système contient **tout ce qu'il faut pour écrire du code
directement utilisable** :

- la **documentation de l'API du jeu** exposée aux scripts : déplacer, viser, jouer une animation,
  jouer un son, infliger des dégâts, créer un projectile, appliquer un effet, lire la position du
  joueur, etc. ;
- les **règles de game design** : budgets de dégâts et de PV par niveau, durées de recharge minimales,
  mécaniques autorisées ;
- des **exemples de scripts** validés ;
- la **fiche** et le **rapport du directeur** du monstre : il sait donc que ce monstre a 8 pattes, une
  queue, une batte dans la main droite, et quelles animations existent vraiment.

### Ce qu'il écrit

Un petit script qui branche les animations, les sons et les règles de combat :

```lua
-- comportement.lua (généré) : Rôdeur des cendres
function on_spawn(self)
  self:play("idle")
end

function on_update(self, dt)
  local cible = self:nearest_player()
  if not cible then return end
  local d = self:distance(cible)
  if d > 8 then
    self:move_towards(cible, self.stats.vitesse)      -- joue "marche" automatiquement
  elseif d > 2 and self:ready("Charge") then
    self:attack("Charge", cible)                      -- animation + dégâts définis dans le manifeste
  elseif d <= 1.5 and self:ready("Morsure") then
    self:attack("Morsure", cible)
  end
end

function on_hit(self, source, degats)
  if self.pv < self.stats.pv * 0.3 then
    self:set_mode("enrage")                           -- plus rapide, mais attaque moins précise
  end
end
```

### Comment on rend ça sûr et fiable : le harnais

Du code écrit par une IA ne doit **jamais** pouvoir casser le jeu, tricher ou attaquer la machine du
joueur. D'où un **harnais** autour du LLM :

1. **Bac à sable** : le script tourne dans un environnement isolé (par exemple Lua sans accès aux
   fichiers ni au réseau, ou WebAssembly) et ne voit **que** l'API du jeu, rien d'autre.
2. **Limites** : temps de calcul par image, mémoire, nombre de projectiles ou d'entités créées.
3. **Validation avant le jeu** : le script doit être syntaxiquement valide, ne citer que des fonctions
   et des animations qui existent, puis passer des **tests automatiques en simulation** sans
   affichage (le monstre attaque-t-il ? peut-il être tué ? reste-t-il dans l'arène ? ses dégâts par
   seconde respectent-ils le budget du niveau ?).
4. **Boucle de correction** : en cas d'échec, l'erreur est renvoyée au LLM, qui corrige (quelques
   essais au maximum).
5. **Secours** : si rien ne passe, le monstre reçoit un **comportement générique** éprouvé, adapté à
   son type de corps. Le joueur a toujours un monstre jouable.
6. **Équilibrage hors du LLM** : les chiffres (PV, dégâts) sont **bornés par le jeu** selon le niveau.
   Le LLM choisit *comment* le monstre se bat, pas *combien* il est puissant au-delà du budget.

Le même mécanisme vaut pour les **objets** : une épée générée arrive avec son script (« ralentit
la cible », « s'enflamme au troisième coup »), validé et borné de la même façon.

## 7. Côté jeu : ce que le moteur doit offrir

Pour que tout cela soit « ultra facile à implémenter », le moteur doit prévoir :

- un **chargeur de paquets** : lit le manifeste, charge le `.glb`, les sons, le script ;
- une **API de script stable et documentée**, la même que celle décrite dans le prompt du LLM. Si
  l'API change, le prompt change avec elle ;
- le **bac à sable** d'exécution ;
- des **comportements génériques** de secours ;
- la **gestion du cycle de vie** : génération à l'avance, utilisation dans le donjon, suppression à
  la sortie.

Le choix du moteur (Godot, Unity, Unreal ou moteur maison) reste ouvert. Le format glTF/GLB et un
langage de script embarquable (Lua, WebAssembly) fonctionnent avec tous.

## 8. Le temps et le calcul : rendre le jetable rentable

Le concept ne tient que si fabriquer un monstre est **rapide et peu coûteux**.

- **Générer à l'avance** : pendant que le joueur explore le donjon N, les monstres du donjon N+1 se
  fabriquent en tâche de fond. Une file d'attente garde quelques monstres d'avance.
- **Mesure actuelle** : environ 20 à 25 minutes par monstre (modèle 3D) sur une RTX 3080 de 10 Go,
  plus l'animation. C'est trop pour du « juste à temps », mais compatible avec une génération
  anticipée, et chaque étape a encore de la marge (masques sur GPU, moins d'essais ratés, modèles plus
  rapides).
- **Où calculer** : sur la machine du joueur (gratuit pour l'éditeur, mais exige une bonne carte
  graphique) ou sur des serveurs (coût par monstre à maîtriser). La pipeline étant 100 % locale
  aujourd'hui, les deux restent possibles.
- **Les pipelines légères** (objets) coûtent bien moins cher : pas d'anatomie, peu ou pas d'animation.

## 9. Points ouverts

- **Budget et format** des modèles pour le jeu (polygones, taille de texture, nombre d'os) : à fixer
  au moment de l'optimisation.
- **Qualité constante sans humain** : c'est l'enjeu n°1. Chaque étape doit savoir détecter seule
  un résultat raté.
- **Cohérence de style** : des monstres uniques, mais qui ont l'air de venir du même jeu (direction
  artistique commune dans le prompt du créateur et des générateurs d'images).
- **Multijoueur** : si plusieurs joueurs partagent un donjon, le paquet doit être envoyé à chacun, et
  le serveur reste seul juge des règles.
- **Licences** des modèles utilisés : à revoir le jour où le jeu sera distribué.

## 10. Où on en est (octobre 2026)

Pipeline monstres, sur une RTX 3080 (10 Go), 100 % en local :

| Étape | État |
|---|---|
| Créateur LLM | à faire : les fiches sont encore écrites à la main |
| Concept (Qwen-Image-2.1) | ✅ validé |
| Vidéo de tour (Wan 2.2 + LoRA orbite) | ✅ validé, montre le dos des corps allongés |
| Choix des vues (VLM + flux optique) | ✅ fonctionne |
| Modèle 3D + texture (argile A2 → Pixal3D 4 vues) | ✅ validé sur 6 morphologies (humanoïde, araignée, loup, limace, Godzilla, objet tenu) |
| Directeur VLM (rapport des membres) | ✅ fonctionne, à fiabiliser sur les corps très symétriques |
| Animation | 🔧 en cours : la marche de profil caméra fixe + R-DMesh donne le meilleur résultat ; test de MoCapAnything V2 (animation de squelette, exportable directement) en cours |
| Sons | à faire |
| Code de comportement + harnais | à faire |
| Optimisation + export jeu | à faire |
| Pipeline objets | à faire |

Prochaines étapes : fiabiliser la marche sur tout le banc d'essai, puis 2-3 monstres inventés par le
LLM, boucle de marche sur place, report sur le modèle texturé et export ; ensuite les attaques, les
sons et le premier code de comportement généré.
