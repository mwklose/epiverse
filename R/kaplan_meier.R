#' Crude Nonparametric Survival function for given data
#'
#' @param data A DataFrame/Tibble with survival data
#' @param ID Identifiers for each individual event
#' @param t The time each atom occurred
#' @param delta 0 is a censoring atom, 1 is an event atom.
#' @param weights What to weight each observation by, for IPW or generalizability weighting
#'
#' @return a Tibble with KM faux hazard and survival at each unique time point.
#' @export
#'
#' @examples kaplan_meier(iosw_data, ID="ID", t="t", delta="delta")
kaplan_meier = function(data, ID, t, delta, weights=NA){
  km = data %>%
    dplyr::select({{ID}}, {{t}}, {{delta}}) %>%
    dplyr::rename(ID = {{ID}}, t = {{t}}, delta = {{delta}})

  times = km %>%
    dplyr::select(t) %>%
    unique() %>%
    dplyr::arrange(t) %>%
    dplyr::group_by(t) %>%# Get distinct time points
    dplyr::summarise(
      n = sum(km[[{{t}}]] >= t), # Get risk set at time points
      events = sum(km[[{{delta}}]] * km[[{{t}}]] == t), # Get events at time points
      faux.hazard = 1 - (events / n) # Get conditional survival at time point
    ) %>%
    dplyr::mutate(
      survival = purrr::accumulate(faux.hazard, `*`) # Accumulate probability of surviving at each time point
    )

  return(times)
}
