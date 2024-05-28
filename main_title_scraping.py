import json
from nesy.dataset_augmentation.title_scraping.amazon_scraper import scrape_title_amazon
from nesy.dataset_augmentation.title_scraping.captcha_scraper import scrape_title_captcha
from nesy.dataset_augmentation.title_scraping.google_scraper import scrape_title_google_search
from nesy.dataset_augmentation.title_scraping.wayback_machine_scraper import scrape_title_wayback


def metadata_scraping(metadata: str,
                      n_cores: int = 1,
                      motivation: str = "no-title",
                      save_tmp: bool = True,
                      batch_size: int = 100,
                      mode: str = None,
                      use_solver: bool = True) -> None:
    """
    This function takes as input a metadata file with some missing titles and uses web scraping to retrieve these
    titles from the Amazon website. It is possible to specify the type of scraping by changing the parameter 'mode'.

    There are four modes currently supported:
        - standard: just looks for missing titles in the Amazon page of the corresponding ASIN;
        - wayback: looks for the same page but in the Wayback machine (note this is not efficient);
        - captcha: looks for titles in the Rocket Source database, which requires solving captchas (a captcha solver
        is required);
        - google: search the ASIN on Google and iters over the results to find the title.

    At the end, a new metadata file with complete information is generated. This metadata file is called
    complete-<input-file-name>.json.

    :param metadata: file containing metadata for the items
    :param n_cores: number of cores to be used for scraping. This has no effect when wayback=True.
    :param motivation: motivation for which the title has to be scraped. It could be:
        - no-title (default): no-title means that the metadata is missing the title for the item. Put
        motivation="no-title" only for the first scraping job, to get the title using the Amazon website
        - captcha-or-DOM: during the first scraping job, the title retrieval for this item failed due to incorrect DOM
        or bot detection from Amazon. If this motivation is selected, the script will take all the ASINs related to this
        error code and will perform another scraping loop on them
        - 404-error: during the first scraping job, the title retrieval failed due to a 404 not found error, meaning the
        ASIN is not on Amazon anymore. If this motivation is selected, the script will take all the ASINs related to
        this error code and will perform a scraping loop using Wayback Machine. Remember to put wayback=True
        - exception-error: in the first scraping job, the title retrieval failed due to an exception
    :param save_tmp: whether temporary retrieved title JSON files have to be saved once the batch is finished
    :param batch_size: number of ASINs that have to be processed for each batch. Keep this number under 100 to avoid
    disk memory problems due to temporary files memorization by Selenium. Keep this number equal to 20 when wayback=True
    :param mode: - "wayback" uses the wayback machine API (for items not found after first scraping loop)
                 - "standard" uses the standard one (for first scraping loop)
                 - "captcha" uses the version that scrapes titles from the Rocket Source knowledge base (to be used
                 after the wayback mode. If even with the Wayback Machine is not possible to get the titles, it is
                 possible to get them from this knowledge base. It is the slowest mode as it requires solving difficult
                 captchas automatically)
                 - "google" uses the Google Search to get the titles corresponding to the given ASINs
    :param use_solver: whether to use a captcha solver if the selected mode is catpcha
    """
    with open(metadata) as json_file:
        m_data = json.load(json_file)
    # take the ASINs for the products that have a missing title in the metadata file
    # only the items with the selected motivation will be picked
    no_titles = [k for k, v in m_data.items() if v == motivation]
    # update the metadata with the scraped titles
    if mode is None or mode == "standard":
        updated_dict = scrape_title_amazon(no_titles, n_cores, batch_size=batch_size, save_tmp=save_tmp)
    elif mode == "wayback":
        updated_dict = scrape_title_wayback(no_titles, batch_size=batch_size, save_tmp=save_tmp)
    elif mode == "captcha":
        updated_dict = scrape_title_captcha(no_titles, n_cores, batch_size=batch_size, save_tmp=save_tmp,
                                            use_solver=use_solver)
    else:
        updated_dict = scrape_title_google_search(no_titles, n_cores, batch_size=batch_size, save_tmp=save_tmp)
    # update of the metadata
    m_data.update(updated_dict)
    # generate the new and complete metadata file
    with open('./data/processed/complete-%s' % (metadata.split("/")[-1]), 'w', encoding='utf-8') as f:
        json.dump(m_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    metadata_scraping("./data/processed/legacy/complete-filtered-metadata.json", motivation="captcha-or-DOM",
                      mode="standard", save_tmp=True, batch_size=100, use_solver=False)
