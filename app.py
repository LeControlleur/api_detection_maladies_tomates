import json
from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2


model = load_model("models/model.h5")
outputs = np.load("variables/dataset_diseases_output.npy")

{
    "0" : {
        "maladie" : "Mildiou de tomates",
        "description": "Le mildiou de la tomate est une maladie cryptogamique, causée par des champignons, Phytophthora infestans et Phytophthora parasitica (mildiou terrestre). La maladie cause le dépérissement des plants de tomates après les pluies d’été et entraine souvent la perte de nombreux fruits.",
        "symptomes": "Mildiou sur tomatesMildiou “aérien” (Phytophthora infestans) : Le champignon attaque d’abord les feuilles. Des taches, d’abord petites, jaunes puis brunes et qui sèchent rapidement apparaissent. Les tiges sont ensuite attaquées et présentent de grandes taches brunes irrégulières. Les fruits à un stade précoce sont également atteints et présentent des marbrures brunes et souvent bosselées très caractéristiques.\nMildiou terrestre (Phytophthora parasitica) : Cette maladie apparaît plus tôt dans la saison que le mildiou aérien. Le premier symptôme est un chancre brun à la base de la tige lorsque la contamination a lieu dès la levée des semis. Le champignon peut également attaquer les fruits sur la face tournée vers le sol, entraînant des confusions avec les maladies de nécrose apicale (“cul noir” associée à une mauvaise assimilation du  calcium souvent liée à une irrigation défectueuse)."
    },
    "1" : {
        "maladie" : "Moisissure des feuilles",
        "description": "Cladosporium fulvum est un ascomycète appelé Passalora fulva, un agent pathogène non obligatoire qui provoque la maladie de la tomate connue sous le nom de moisissure des feuilles de la tomate. P. fulva n'attaque que les plants de tomates, en particulier le feuillage, et c'est une maladie courante dans les serres, mais peut également se produire dans les champs. [2] L'agent pathogène est susceptible de se développer dans des conditions humides et fraîches.",
        "symptomes": "Les symptômes de la moisissure des feuilles de tomate apparaissent généralement avec le feuillage, mais l'infection des fruits est rare. Le symptôme primaire apparaît sur la surface supérieure des feuilles infectées sous la forme d'une petite tache vert pâle ou jaunâtre avec des bords indéfinis, et sur la zone correspondante de la surface inférieure, le champignon commence à sporuler. Le symptôme diagnostique se développe sur la surface inférieure sous la forme d'un aspect vert olive à violet grisâtre et velouté, composé de spores ( conidies ). En continu, la couleur de la feuille infectée vire au brun jaunâtre et la feuille commence à s'enrouler et à sécher. Les feuilles tomberont en atteignant un stade prématuré et la défoliation de l'hôte infecté provoquera une infection supplémentaire. Cette maladie se développe bien entaux d'humidité supérieurs à 85 %."
    },
    "2" : {
        "maladie" : "Le virus de la mosaïque de la tomate",
        "description": "Le virus de la mosaïque de la tomate est un phytovirus pathogène, qui affecte principalement les cultures de tomate. Il est classé dans le genre des Tobamovirus, et apparenté au virus de la mosaïque du tabac qui est l'espèce-type de ce genre.",
        "symptomes": "Le feuillage des plants de tomates affectés présente des marbrures, avec une alternance de zones jaunâtres et vertes plus foncées, ces dernières apparaissant souvent plus épaisses et surélevées donnant un aspect ressemblant à des cloques. Les feuilles ont tendance à ressembler à des fougères avec des extrémités pointues et les jeunes feuilles peuvent être tordues. Le fruit peut être déformé, des taches jaunes et des taches nécrotiques peuvent apparaître sur les fruits mûrs et verts et il peut y avoir un brunissement interne de la paroi du fruit. Chez les jeunes plants, l'infection réduit la nouaison et peut provoquer des déformations et des imperfections. La plante entière peut être rabougrie et les fleurs décolorées."
    },
    "3" : {
        "maladie" : "Virus Corynespora cassiicola",
        "description": "La maladie survient sur les tomates cultivées en plein champ dans les régions tropicales et subtropicales du monde. Les infections au virus Corynespora cassiicola réduisent le rendement indirectement en réduisant la zone photosynthétique et directement en réduisant la valeur marchande du fruit à travers les points de fruits.",
        "symptomes": "Cette maladie se manifeste par des petites lésions humides apparaissant à la face supérieure des feuilles, parfois localement limitées par une nervure. I peut aussi apparaître des taches plutôt circulaires et atteignant 2 cm de diamètre, auréolées d’un halo jaune bien visible, et présentant des motifs concentriques rappellent ceux d’une cible (target spots). Il y a aussi des lésions brunes et longitudinales sur tige et pétioles, ceinturant parfois entièrement celle-ci et engendrant le dessèchement de feuilles. Petites taches sur les fleurs et surtout les fruits, brun clair avec une marge plus foncée, et de consistance sèche. Elles sont plus larges et circulaires sur les fruits mûrs. Légèrement déprimées, elles finissent par brunir et se fendre en leur centre."
    },
    "4" : {
        "maladie" : "Le tétranyque à deux points",
        "description": "Le tétranyque à deux points est l'espèce d'acarien la plus commune qui attaque les cultures maraîchères et fruitières",
        "symptomes": "Au champ, les tétranyques sont favorisés par un temps chaud et sec, qui aggrave également les blessures en stressant la plante. Les dégâts sont souvent sous-estimés car les blessures et le ravageur ne sont pas apparents à nos yeux sans une inspection minutieuse. Les feuilles se tachent de taches jaune pâle et brun rougeâtre allant de petites à grandes surfaces sur les surfaces supérieure et inférieure des feuilles. D'autres symptômes causés par une attaque grave ou constante comprennent des feuilles déformées, une perte globale de vigueur de la plante (malgré une humidité et une nutrition adéquates), le blanchissement ou la tache des feuilles, le jaunissement de la plante ou de certaines feuilles et, dans certains cas, la perte de feuillage et mort."
    },
    "5" : {
        "maladie" : "Virus des feuilles jaunes en cuillère de la tomate",
        "description": "Le Virus des feuilles jaunes en cuillère de la tomate (Tomato yellow leaf curl virus - TYLCV) est une espèce de phytovirus du genre Begomovirus (famille des Geminiviridae) responsable d'une des principales maladies virales des cultures de tomates. Ce virus affecte également diverses plantes hôtes, dont plusieurs plantes maraîchères (piment, haricot, tabac...) ainsi que diverses adventices.",
        "symptomes": "Les symptômes sont un jaunissement des feuilles, dont les folioles s'incurvent vers le haut (« en cuillère »), et une forte réduction de la croissance si l'infection est grave."
    },
    "6" : {
        "maladie" : "Mildiou de pommes de terre",
        "description": "Le mildiou est causé par l'oomycète Phytophthora infestans . Les oomycètes sont des organismes ressemblant à des champignons, également appelés moisissures aquatiques, mais ce ne sont pas de vrais champignons.",
        "symptomes": "Les feuilles ont de grandes taches brun foncé avec un bord gris vert; non confiné par les nervures principales des feuilles. Les infections progressent à travers les folioles et les pétioles, ce qui donne de grandes sections de feuillage brun sec. Les infections des tiges sont fermes et brun foncé avec un bord arrondi. Des taches circulaires fermes, brun foncé, recouvrent de grandes parties des fruits. Les taches peuvent devenir pâteuses à mesure que des bactéries secondaires envahissent.En cas d'humidité élevée, une fine croissance fongique blanche poudreuse apparaît sur les feuilles, les fruits et les tiges infectés. Par temps frais et humide, des champs entiers brunissent et se fanent comme s'ils étaient frappés par le gel."
    },
    "7" : {
        "maladie" : "La tache bactérienne",
        "description": "La tache bactérienne de la tomate est une maladie potentiellement dévastatrice qui, dans les cas graves, peut entraîner des fruits non commercialisables et même la mort des plantes. La tache bactérienne peut se produire partout où les tomates sont cultivées, mais se trouve le plus souvent dans les climats chauds et humides, ainsi que dans les serres.",
        "symptomes": "La tache bactérienne peut affecter toutes les parties aériennes d'un plant de tomate, y compris les feuilles, les tiges et les fruits. La tache bactérienne apparaît sur les feuilles sous forme de petites zones circulaires (moins de ⅛ de pouce), parfois imbibées d'eau (c'est-à-dire d'apparence humide). Les taches peuvent d'abord être jaune-vert, mais s'assombrir à rouge-brun en vieillissant. Lorsque la maladie est grave, un jaunissement important des feuilles et une perte de feuilles peuvent également survenir. Sur les fruits verts, les taches sont généralement petites, surélevées et ressemblant à des cloques, et peuvent avoir un halo jaunâtre. À mesure que le fruit mûrit, les taches s'agrandissent et deviennent brunes, croûteuses et rugueuses. Les taches matures peuvent être surélevées ou enfoncées avec des bords surélevés."
    },
    "8" : {
        "maladie" : "La tache septorienne",
        "description": "La tache septorienne est causée par un champignon, Septoria lycopersici . C'est l'une des maladies les plus destructrices du feuillage de la tomate et elle est particulièrement grave dans les régions où le temps humide persiste pendant de longues périodes.",
        "symptomes": "La tache septorienne apparaît généralement sur les feuilles inférieures après la première nouaison. Les taches sont circulaires, d'environ 1/16 à 1/4 de pouce de diamètre avec des marges brun foncé et des centres beiges à gris avec de petites structures de fructification noires. De manière caractéristique, il y a beaucoup de taches par feuille. Cette maladie se propage vers le haut de la croissance la plus ancienne à la plus jeune. Si les lésions foliaires sont nombreuses, les feuilles jaunissent légèrement, puis brunissent, puis se fanent. L'infection des fruits est rare."
    },
    "9" : {
        "maladie" : "saine",
        "description": "",
        "symptomes": ""
    }
}

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def upload_file():
    #   Enregistrement du fichier
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save("files/"+uploaded_file.filename)

        y_pred = prediction(uploaded_file.filename)
        print(outputs)
        y_pred = str(y_pred)
        
        with open("variables/diseases.json") as file:
            diseases_dict = json.load(file)
            print("\n\nLa maladie de la plante est : {} \n\n".format(y_pred))
            print(diseases_dict[y_pred])


    #   Prédiction à partir de l'image

    else :
        return 

    return redirect(url_for('index'))



def prediction(filename):
    filepath = os.path.join("files", filename)

    image = cv2.imread(filepath)
    image = np.expand_dims(image, 0)
    y_pred = model.predict(image)
    #print(y_pred)
    y_pred = np.argmax(y_pred)
    #print("\n\nLA PREDICTION EST : ", y_pred)

    return y_pred 
    

app.run()