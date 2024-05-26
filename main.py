# STATISTICS METADATA
# start: 363876 matched and 116159 without title
# first scraping: 456341 matched and 441 are captcha-or-DOM and 23253 are 404 error
# end: 467833 matched and 11723 are 404 error and 479 are captcha in wayback -> captcha are more because some of the
# 404 error before became a captcha after Wayback Machine. Note also that about 70 items with a DOM problem have to be
# manually inserted in the file because Soup was not finding an ID that was actually there
# ALCUNE STATISTICHE
# metadati: 95% completi di titolo, 4.5% non hanno il titolo (404 error), 1% problemi di DOM o captcha
# musica: 72% (46238) item matchati su wikidata
# musica (senza film): 67% (43392) item matchati su wikidata
# movies: 60% (30101) item matchati su wikidata
# books: 16% (57946) item matchati su wikidata
# interessante perche' nonostante i libri siano solo il 16%, il dataset e' comunque piu' completo degli altri due, quindi nessun problema, possiamo utilizzarlo
# todo vedere discorso delle descrizioni dei film, libri ecc in altri database, per calcolare similarita' tra loro


# todo INTERESSANTE
# due concetti:
# 1: trovare uno score di similarita' per capire se utilizzare quel cross-domain path per fare raccomandazione o meno.
# avra' uno score tra 0 e 1 da inserire nella regola logica
# come calcolare lo score per questa coppia di item cross-domain? numero di path, lunghezza dei path, embedding dei singoli item all'interno del path e cosi via
# 2: fare ranking dei path in fase di inferenza per decidere quale path utilizzare per la explanation finale

# ogni scalino quando arriva.
# 1 BASELINE: potrebbe essere di semplicemente non utilizzare nessun approccio e fare cross-domain recommendation basata solo sul cammino
# utile sia in fase si training per fare regolarizzazione in caso di dati sparsi sul target domain, ma anche in fase di inferenza per risolvere i casi di cold-start, perche' possiamo usare direttamente i cammini
# una prova versione potrebbe utilizzare solamente item che sono stati matchati su wikidata, una seconda versione potrebbe comprendere tutti gli item del dataset, e utilizzare quelli matchati per trasferire informazione
# vogliamo spiegare principalmente il cold-start e li diventa essenziale sfruttare la informazione che lega i due domini

# todo appunti di Alessio
# Evaluating the reliability of LLMs is complex. Works such as https://arxiv.org/abs/2401.00761 and https://www.sciencedirect.com/science/article/pii/S266734522300024X clearly show that using LLMs as knowledge bases is
# dangerous. Most of question answering results reported in literature are probably over estimated due to data contamination, and LLMs are known to be prone to hallucinations, especially for specific domain knowledge.
# I'm not sure whether accessing existing books protected by copyright is ethical (or should even be legal). Just because ChatGPT is able to do so, that doesn't mean users should be allowed to or reuse the content
# for (possibly) commercial content.
# https://obsidian.md/
# https://segment-anything.com/

# todo cose interessanti e gia' disponibili
# kgtk user interface for knowledge graph embeddings: https://kgtk.isi.edu/similarity/
# kgtk code for path finding in Wikidata: find under similarity module (kgtk-similarity repo) -> we need to understand why it is deprecated
# chatGPT or LLMs in general to compansate for errors in the KG (KG completion problem) -> hallucinations, ethical issues? Find papers
# https://github.com/RManLuo/Awesome-LLM-KG
# https://arxiv.org/abs/2310.06671

# todo conversazione con Nicolo'
# OFF-TOPIC ci serve un sistema di raccomandazione super personalizzato, ad esempio, e se volessi solo film che hanno un voto maggiore di X su iMDB? Come faccio a chiedere una cosa simile?
# per avere spiegazioni veramente personalizzate per gli utenti, qualcosa bisogna apprendere sugli utenti, bisogna apprendere come effettuare le spiegazioni. Ci sono degli aspetti che agli utenti interessano e degli aspetti che non interessano. Noi dobbiamo cercare di capire quali questi aspetti siano e spiegare
# quali sono i pro di post-hoc o intrinsic? Sono le post-hoc più personalizzate anche se non riflettono le decisioni del modello?
# Anche feature-based potrebbe essere interessante. Alla fine questa cosa di iMDB potrebbe essere una latent feature
# mapping tra datasets e KGs https://github.com/RUCDM/KB4Rec -> vedere anche il paper
# E' figo perche' si puo' direttamente scaricare il dump che ci interessa a noi date le triple
# Tuttavia, i loro dataset non ci sono utili perché non hanno utenti in comune, ma l'approccio che hanno utilizzato potrebbe esserci molto utile per cercare di fare una cosa simile
# il mapping è comunque minimo come nel nostro caso, quindi possiamo citare in caso e dire che comunque è challenging fare questi mappings

# https://drive.google.com/drive/folders/18pEKcUSWt0uFGqDukk6pcP0RASxuexYr?usp=drive_link tutorial su KG recommendation
# come ti dicevo, nessuno controlla se i match sono validi -> controllarli sarebbe un contributo
# paper sul mapping https://direct.mit.edu/dint/article/1/2/121/27497/KB4Rec-A-Data-Set-for-Linking-Knowledge-Bases-with
# If no KB entity with the exact same title was returned, we say the RS item is rejected in the linkage process. -> il loro funziona solo se il match e' esatto
# il nostro funziona anche con match non esatti e lingue diverse, grazie all'algoritmo intelligente di Wikidata Special Search
# We have found only a small number (about 1,000 for each domain) of RS items cannot be accurately linked or rejected via the above procedure, and we simply discard them.
# interessante che solo 1000 item in ogni dominio sono stati scartati, quindi significa che funziona benone, o forse sono domini piccoli e 1000 item sono molti
# We find that most of the linkage in LFM-1b and Amazon book data sets can be determined accurately (either linked or non-linked) in this way.
# interessante anche questa osservazione
# Per spiegare, possiamo anche fornire i path ad un LLM, che deve basare il testo sul Path. Così sappiamo che la spiegazione è giusta, ma è fornita in un una maniera tale che diventa più enjoyable per l'utente
# Pearlm and PGPR mapper??

# INTERESSANTE -> Sto pensando magari potrebbe essere interessante esplorare quali relazioni ci siano tra le preferenze degli utenti e i rating della critica. Secondo me ci sono utenti a cui piacciono proprio i film snobbati dalla critica
# Allucinazioni: e se sei attratto dalla descrizione del film prodotta dal modello e poi ti accorgi che il film parla di tutt'altro?
# Seconda: quanto figo sarebbe parlare di group recommendation o context aware recommendation con gli LLM?

# Quindi dato un utente nel target domain, consideriamo gli item a cui l'utente non ha dato rating. Per tutti questi item andiamo a cercare cammini che li collegano ad item che sappiamo l'utente apprezza nel source domain. Infine se i due item hanno una scarsa similarità allora scartiamo i cammini e facciamo inferenza col modello del target, altrimenti raccomandiamo gli item che matchano
# Bisogna trovare il prompt giusto e la maniera giusta di interpretarlo
# Sicuramente serve imporre una struttura nelle risposte, altrimenti serve un modello di NLP solo per estrarre i dati. Poi non so quanto questa venga rispettata
# Questo può funzionare, però penso che oltre alle allucinazioni l'ostacolo maggiore sia che può tirare fuori di tutto

# todo verbale dell'ultima call
# chatGPT sembra essere diventato promettente, grazie a dei prompt ingiegnerizzati
# kgtk abbiamo inizato l'installazione e abbiamo fatto partire i primi job
# discorso path sembra esserci tutto direttamente su kgtk con le API da terminale -> capire se sono meglio queste o quelle dentro kgtk-similarity
# discorso componenti connesse ma forse piu' semplice fare con reachable-nodes API
# PoC -> troviamo i path e se esiste usiamo chatGPT per capire la plausibilita', doppio checker, similarity di KGTK (metrica migliore e' la complex) + chatGPT

import json
import os

import pandas as pd

from nesy.data import (
    create_asin_metadata_json,
    create_pandas_dataset,
    entity_linker_api,
    entity_linker_api_query,
    filter_metadata,
    get_wid_per_cat,
    metadata_cleaning,
    metadata_scraping,
    metadata_stats,
    get_cross_pairs,
    remove_movies_from_music,
    split_metadata,
    entity_linker_title_person_year,
)
from nesy.paths import get_multiple_paths, get_paths
from nesy.paths.merge_tsv_files import merge_tsv_from_directory
from nesy.paths.labels import generate_all_labels
from nesy.preprocess_kg import preprocess_kg
from nesy.dataset_augmentation.utils import correct_missing_types, get_metadata_stats

if __name__ == "__main__":
    correct_missing_types("./data/processed/merged_metadata.json")
    get_metadata_stats("./data/processed/merged_metadata.json")
    # correct_missing_types("./data/processed/merged_metadata.json")
    # entity_linker_title_person_year("./prova.json")
    # metadata_stats("./data/processed/complete-filtered-metadata.json",
    #                errors=["404-error"], save_asins=True)
    # metadata_stats("./data/processed/mapping-reviews_CDs_and_Vinyl_5.json",
    #                errors=["not-in-dump", "not-found-query", "not-title"], save_asins=False)
    # split_metadata("./data/processed/final-metadata.json")
    exit()
    # generate_all_labels("./data/paths")
    # merge_tsv_from_directory(
    #     "./data/paths/Q103474-Q482621", "./data/paths/Q103474-Q482621/paths_all.tsv"
    # )
    # convert_ids_to_labels("./data/paths/Q103474-Q482621/paths_all.tsv")
    kg = "./data/wikidata/claims.wikibase-item_preprocessed.tsv.gz"
    cache = "./data/wikidata/graph-cache.sqlite3.db"
    # selected_relations = pd.read_csv("./data/wikidata/selected-relations.csv")[
    #     "ID"
    # ].tolist()
    # preprocess_kg(
    #     input_graph=kg,
    #     cache_path=cache,
    #     compress_inter_steps=False,
    #     debug=True,
    #     selected_properties=selected_relations,
    # )
    # kg_preprocessed = "./data/wikidata/claims.wikibase-item_preprocessed.tsv.gz"
    # get_paths(
    #     input_graph=kg_preprocessed,
    #     graph_cache=cache,
    #     output_dir="data/paths",
    #     source="Q3906523",
    #     target="Q22000542",
    #     max_hops=3,
    #     debug=True,
    # )
    # pairs = [
    #     # 2001: A Space Odyssey -> The Blue Danube
    #     ("Q103474", "Q482621"),
    #     # Waldmeister -> 2001: A Space Odyssey
    #     ("Q7961534", "Q103474"),
    #     # The Rains of Castamere -> Game of Thrones
    #     ("Q18463992", "Q23572"),
    #     # Do Androids Dream of Electric Sheep? -> Blade Runner 2049
    #     ("Q605249", "Q21500755"),
    #     # Ready Player One (book) -> Ready Player One (film)
    #     ("Q3906523", "Q22000542"),
    #     # The Lord of the Rings: The Two Towers -> The Hobbit (book)
    #     ("Q164963", "Q74287"),
    #     # American Pie Presents: Band Camp -> The Anthem
    #     ("Q261044", "Q3501212"),
    #     # New Divide -> Transformers
    #     ("Q19985", "Q171453"),
    #     # Halloween -> Dragula
    #     ("Q909063", "Q734624"),
    #     # Timeline -> Jurassic Park
    #     ("Q732060", "Q167726"),
    #     # My Heart Will Go On -> Inception
    #     ("Q155577", "Q25188"),
    #     # The Godfather -> The Sicilian
    #     ("Q47703", "Q960155"),
    #     # The Girl with the Dragon Tattoo (podcast episode) - > The Girl Who Played with Fire (book)
    #     ("Q116783360", "Q1137369"),
    # ]
    gen, gen_len = get_cross_pairs("music", "movies")
    get_multiple_paths(
        input_graph=kg,
        graph_cache=cache,
        output_dir="data/paths",
        pairs=gen,
        max_hops=2,
        debug=False,
        n_jobs=6,
        gen_len=gen_len,
    )
    # create_wikidata_labels_sqlite("./data/wikidata/labels.en.tsv")
    # convert_ids_to_labels("./data/wikidata/results/Q103474-Q482621/query_results_2_hops.tsv")
    # todo prendere descrizione film, libri e musica -> segnale semantico + segnale latente -> si possono vedere le varie visioni
    # todo ASIN dei libri coincide con ISBN, ma tanto non abbiamo neanche quello
    # todo esperimenti sia su globale che solo su quelli matchati
    # filter_metadata("./data/processed/metadata.json", ["reviews_Movies_and_TV_5",
    #                                                    "reviews_Books_5",
    #                                                    "reviews_CDs_and_Vinyl_5"])
    # create_asin_metadata_json("./data/raw/metadata.json")
    # metadata_scraping("./data/processed/metadata.json")
    # create_pandas_dataset("./data/raw/reviews_Books_5.json")
    # entity_linker_api_query("./data/processed/reviews_Books_5.csv", use_dump=True)
    # get_wid_labels("./data/raw/movies.json")
    # get_wid_labels("./data/raw/music.json")
    # get_wid_labels("./data/raw/books.json")
    # get_wid_per_cat("music")
    # metadata_scraping("./data/processed/complete-filtered-metadata.json", motivation="404-error", save_tmp=True,
    #                   batch_size=100, mode="search")
    # entity_linker_api_query("./data/processed/reviews_CDs_and_Vinyl_5.csv", use_dump=True, retry=True, retry_reason="not-title")
    # metadata_scraping("./data/processed/final-metadata.json", 1,
    #                   motivation="DOM", save_tmp=True, batch_size=20, wayback=True)
    # metadata_stats("./data/processed/mapping-reviews_Movies_and_TV_5.json",
    #                errors=["not-in-dump", "not-found-query", "not-title"], save_asins=False)
    # metadata_stats("./data/processed/mapping-reviews_CDs_and_Vinyl_5.json",
    #                errors=["not-in-dump", "not-found-query", "not-title"], save_asins=False)
    # metadata_stats("./data/processed/mapping-reviews_Books_5.json",
    #                errors=["not-in-dump", "not-found-query", "not-title", "exception"], save_asins=False)
    # metadata_stats("./data/processed/complete-filtered-metadata.json", errors=["captcha-or-DOM", "404-error"], save_asins=False)
    # metadata_stats("./data/processed/final-metadata.json", errors=["404-error", "DOM", "captcha"], save_asins=False)
    # metadata_stats("./data/processed/mapping-reviews_Movies_and_TV_5.json", errors=["not-found", "not-in-dump"], save_asins=False)

    # get_wid_per_cat("books")
    # metadata_cleaning("./data/processed/complete-filtered-metadata.json")
    # with open("./data/processed/mapping-reviews_Movies_and_TV_5.json") as json_file:
    #     mapping = json.load(json_file)
    # p = pd.read_csv("./data/processed/reviews_Movies_and_TV_5.csv")
    # print(p["itemId"].nunique())
    # print(len(mapping))
    # print((len([m for m in mapping if mapping[m] != ""])))
    # with open("./data/processed/complete-filtered-metadata.json") as json_file:
    #     m_data = json.load(json_file)
    # # take the ASINs for the products that have a missing title in the metadata file
    # no_titles = [k for k, v in m_data.items() if v == "404-error"]
    # # collect_wayback_links(no_titles[:100], os.cpu_count(), 100)
    # scrape_title_wayback_api_2(no_titles[:100], 1, 1, False)

    # with open("./data/processed/complete-filtered-metadata.json") as json_file:
    #     m_data = json.load(json_file)
    # for filename in os.listdir("./data/processed"):
    #     f = os.path.join("./data/processed", filename)
    #     # checking if it is a file
    #     if os.path.isfile(f) and f[-1].isdigit():
    #         with open(f) as json_file:
    #             content = json.load(json_file)
    #             m_data.update(content)
    # with open('./data/processed/final-metadata.json', 'w', encoding='utf-8') as f:
    #     json.dump(m_data, f, ensure_ascii=False, indent=4)
