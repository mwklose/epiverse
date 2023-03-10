% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/kaplan_meier.R
\name{kaplan_meier}
\alias{kaplan_meier}
\title{Crude Nonparametric Survival function for given data}
\usage{
kaplan_meier(data, ID, t, delta, weights = NA)
}
\arguments{
\item{data}{A DataFrame/Tibble with survival data}

\item{ID}{Identifiers for each individual event}

\item{t}{The time each atom occurred}

\item{delta}{0 is a censoring atom, 1 is an event atom.}

\item{weights}{What to weight each observation by, for IPW or generalizability weighting}
}
\value{
a Tibble with KM faux hazard and survival at each unique time point.
}
\description{
Crude Nonparametric Survival function for given data
}
\examples{
kaplan_meier(iosw_data, ID="ID", t="t", delta="delta")
}
