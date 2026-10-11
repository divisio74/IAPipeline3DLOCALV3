#!/usr/bin/env python
"""allure_marche.py — gabarit WALK : l'ALLURE est écrite par le modèle de langage à partir du RAPPORT DU DIRECTEUR
(2026-10-01). Rien de codé par type de créature : le modèle reçoit la description de la fiche, le rapport (membres,
exemplaires, articulé/bloc, objets) et la vue de profil, et décrit comment CE corps marche.
LLM provisoire = Qwen3-VL-4B (déjà utilisé par le directeur) ; le futur « créateur » le remplacera.
Usage (⚠ .venv-trellis2) : python bench/allure_marche.py <id ex. v3_quadrupede>
Sortie : bench/out/<id>/marche_allure.txt (une à trois phrases en anglais, à insérer dans le prompt vidéo)"""
import json
import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from directeur2 import VLM  # noqa: E402

B = Path(__file__).resolve().parent
CHIFFRES = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def comptes(O):
    """nombre d'exemplaires de chaque membre = maximum de cadres posés par le VLM du directeur sur une des 4 vues
    (le rapport, lui, découpe un bras en plusieurs morceaux). Indicatif : se trompe quand des membres se cachent."""
    import ast
    r = {}
    for e in json.load(open(O / "directeur2" / "elements.json", encoding="utf-8"))["elements"]:
        if e["genre"] != "membre":
            continue
        c = e["cadres"] if isinstance(e["cadres"], dict) else ast.literal_eval(e["cadres"])
        r[e["nom"]] = max([len(b) for b in c.values()] + [1])
    return r


NUM = r"(one|two|three|four|five|six|seven|eight|\d)"


def nombres_description(desc, noms):
    """{partie: nombre} lus dans la description (« three-headed », « four strong clawed legs », « two large leathery
    bat-like wings ») : chiffre suivi de 0 à 3 mots puis du nom de la partie (pluriel ou adjectif en -ed)"""
    inv = {v: k for k, v in CHIFFRES.items()}
    r = {}
    for m in re.finditer(r"\b" + NUM + r"\b", desc, re.I):
        # groupe nominal qui suit le nombre (jusqu'à 4 mots, avant virgule / « with » / « and ») : le nombre porte
        # sur le DERNIER nom de partie (« four strong clawed legs » -> legs, pas claw)
        suite = re.split(r",|;|\bwith\b|\band\b|\bon\b", desc[m.end():], maxsplit=1, flags=re.I)[0]
        mots = re.findall(r"[\w-]+", suite)[:4]                       # « bat-like » = un seul mot
        trouves = [nom for w in mots for nom in noms if w.strip("-").lower().startswith(nom)]
        if trouves:
            w = m.group(1).lower()
            r[trouves[-1]] = int(w) if w.isdigit() else inv[w]
    return r


def manquants(texte, attendus):
    """parties (nombre >= 2) dont l'allure ne cite pas le nombre : « its four arms », « three snarling heads »"""
    ordinaux = {2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth"}
    return [f"{CHIFFRES.get(n, n)} {nom}s" for nom, n in attendus.items() if n >= 2 and not re.search(
        r"\b(" + CHIFFRES.get(n, str(n)) + r"|" + str(n) + r")(?:[\s-]+[\w-]+?){0,3}?[\s-]+" + nom, texte, re.I)
        and not re.search(r"\b" + ordinaux.get(n, "#") + r"[\s-]+" + nom, texte, re.I)]   # « its fourth arm » : ok


def main():
    ident = sys.argv[1]
    # --ample : chaque partie (têtes, queue, ailes, bras...) fait un mouvement AMPLE et bien visible pendant la marche.
    # La consigne d'origine demandait « keeping them moving gently » -> parties secondaires quasi figées dans Wan,
    # donc absentes de l'animation (Cerbère : têtes immobiles, remarque du user 2026-10-07).
    ample = "--ample" in sys.argv[2:]
    O = B / "out" / ident
    job = json.load(open(B / "jobs" / f"{ident}.json", encoding="utf-8"))
    rapport = (O / "directeur2" / "rapport.txt").read_text(encoding="utf-8")
    vue = Image.open(O / "directeur2" / "vue_gauche.png").convert("RGB")
    # NOMBRES EXACTS (2026-10-10) : l'allure ample de Goro (4 bras) ne parlait que de « right arm / left arm » ->
    # Wan a animé un personnage à 2 bras (bras fusionnés/disparus). Chaque groupe de parties doit être nommé avec son
    # nombre : celui de la DESCRIPTION en priorité, sinon le comptage des cadres du directeur.
    cpt = comptes(O)
    liste = ", ".join(f"{CHIFFRES.get(n, n)} {nom}{'s' if n > 1 else ''}" for nom, n in cpt.items())
    q = (f"This image shows a 3D game monster from the side. Its description: \"{job['desc_courte']}\".\n"
         f"An analysis of its 3D model found these body parts:\n{rapport}\n"
         f"Counting on four views of the model (approximate, hidden parts can be missed): {liste}.\n\n"
         "EXACT COUNTS: use the number of each body part stated in the description; when the description gives "
         "no number, use the counting above. Always name each group of parts WITH its number (for example \"its "
         "four arms\", \"its three heads\", \"its two wings\"), describe every one of them, and never describe only "
         "a \"left\" and a \"right\" one when it has more than two: no part may be merged, removed or added.\n\n"
         "Write how THIS creature walks forward at a steady pace, as it would in a video game walk cycle, using "
         "only the body parts it really has. In " + ("3 or 4" if ample else "2 or 3") + " short sentences, say: which limbs it walks on and in what "
         "order they step (a natural gait for its anatomy), how its body moves while walking (bobbing, swaying), "
         + ("and, for EACH of its other body parts (every head, neck, tail, wing, arm, tentacle, held object), one "
            "large, clearly visible movement it makes at the same time (turning, swinging, snapping, looking around, "
            "flapping), with big amplitude: nothing stays still. When it has several similar parts (several "
            "heads, arms, wings or tentacles), each of them moves INDEPENDENTLY with its own different movement and "
            "timing, never in unison. "
            if ample else
            "and what its other parts do (tail, tentacles, head, held objects), keeping them moving gently. ")
         + "It walks continuously with regular steps, no running, no jumping, no stopping, no turning. Do not mention "
         "the camera or the background. Answer with the sentences only, starting with \"It walks\".")
    vlm = VLM()
    # vérification : chaque nombre lu dans la description doit être cité ; sinon on redemande en signalant l'oubli
    # (Qwen3-VL-4B a écrit « its head » pour Cerbère, « two strong legs » pour le dragon à quatre pattes)
    attendus = nombres_description(job["desc_courte"], [n for n in cpt if n != "torso"])
    qe = q
    for essai in range(4):
        rep = re.sub(r"\s+", " ", vlm([vue], qe, n=260 if ample else 180)).strip().strip('"')
        oubli = manquants(rep, attendus)
        print(f"[allure] essai {essai + 1} : {rep}" + (f"\n[allure] -> manque : {', '.join(oubli)}" if oubli else ""),
              flush=True)
        if not oubli:
            break
        qe = (q + f"\n\nA previous answer was WRONG: \"{rep}\"\nIt forgot that this creature has "
              + ", ".join(oubli) + ". Write the answer again, naming " + ", ".join(oubli)
              + " explicitly with their number and describing each of them.")
    # --sortie <fichier> : écrire ailleurs (test sans écraser l'allure en service)
    sortie = sys.argv[sys.argv.index("--sortie") + 1] if "--sortie" in sys.argv else None
    (Path(sortie) if sortie else O / ("marche_allure_ample.txt" if ample else "marche_allure.txt")).write_text(rep, encoding="utf-8")
    print("[allure]", rep, flush=True)


if __name__ == "__main__":
    main()
