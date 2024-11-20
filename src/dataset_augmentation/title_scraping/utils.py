import json


def metadata_stats(metadata, errors, save_asins=False):
    """
    This function produces some statistics for the provided metadata file. The statistics include the number of items
    with a missing title due to a 404 error, a bot detection, DOM error, or an unknown error due to an exception in
    the scraping procedure. Statistics also include the number of items for which person (director, author, or artist)
    and release date are missing. This utility function is useful to have an idea of how much information is still
    missing after scraping.

    The parameter `errors` allows to define which statistics to include in the output, chosen from (404-error,
    captcha-or-DOM, exception-error).
    If save_asins is set to True, for each of these statistics, the output will contain the set of ASINs corresponding
    to each error.
    The output will also contain the percentage of items included in each statistics given the total
    number of items in the provided file.
    The output will be saved to a JSON file in the same location of the given metadata file with "_stats" added at
    the name. This, if save_asins is set to True, otherwise it just prints on the standard output.

    :param metadata: path to the metadata file for which the statistics have to be generated
    :param errors: list of strings containing the name of the errors that have to be included in the statistics. The
    script will search for these specific names in the values of the metadata, for each of the ASINs
    :param save_asins: whether to save the set of the ASINs for each statistic or just produce the statistic
    """
    link_prefix = "https://www.amazon.com/dp/"
    errors = {e: {"counter": 0, "asins": []} for e in errors}
    with open(metadata) as json_file:
        m_data = json.load(json_file)
    for asin, data in m_data.items():
        if type(data) is not dict:
            if data in errors:
                errors[data]["counter"] += 1
                if save_asins:
                    errors[data]["asins"].append(link_prefix + asin)
        else:
            if "matched" in errors:
                errors["matched"]["counter"] += 1
            else:
                errors["matched"] = {"counter": 1}

            if data["person"] is None:
                if "person" not in errors:
                    errors["person"] = {"counter": 1}
                    if save_asins:
                        errors["person"]["asins"] = [link_prefix + asin]
                else:
                    errors["person"]["counter"] += 1
                    if save_asins:
                        errors["person"]["asins"].append(link_prefix + asin)

            if data["year"] is None:
                if "year" not in errors:
                    errors["year"] = {"counter": 1}
                    if save_asins:
                        errors["year"]["asins"] = [link_prefix + asin]
                else:
                    errors["year"]["counter"] += 1
                    if save_asins:
                        errors["year"]["asins"].append(link_prefix + asin)

            if data["year"] is None and data["person"] is None:
                if "person+year" not in errors:
                    errors["person+year"] = {"counter": 1}
                    if save_asins:
                        errors["person+year"]["asins"] = [link_prefix + asin]
                else:
                    errors["person+year"]["counter"] += 1
                    if save_asins:
                        errors["person+year"]["asins"].append(link_prefix + asin)

    if save_asins:
        with open(metadata[:-5] + "_stats.json", "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=4)
    else:
        print(errors)
