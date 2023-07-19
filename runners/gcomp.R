library(tidyverse)

data = read_csv("./Epiverse/epiverse/data/data.csv")


# Outcome models: the easy part
outcome_model = glm(Y3 ~ A2 * W2 * A1 * W1, data=data, family=binomial())
outcome_model |> summary()

predicted_mean = predict(outcome_model, type="response")

unique_data = data |> select(-matches("Y")) |> unique()

unique_data |>
  mutate(
    predicted_unique_data = predict(outcome_model, newdata=unique_data, type="response")
  )


# Density Models: the less easy part
density_k0 = glm(W0 ~ 1, data=data, family=binomial())

density_k1 = glm(W1 ~ A0 * W0, data=data, family=binomial())

density_k2 = glm(W2 ~ A1 * W1 * A0 * W0, data=data, family=binomial())


# Treatment versions:
# - never treat
# - always treat
# - natural course

never_treat = unique_data |>
  mutate(
    A0 = 0,
    A1 = 0,
    A2 = 0
  )

always_treat = unique_data |>
  mutate(
    A0 = 1,
    A1 = 1,
    A2 = 1
  )

gcomp_multiple_time_points = function(dataset){
  dataset |>
    mutate(
      predicted_outcome = predict(outcome_model, newdata=dataset, type="response"),
      predicted_density_k0 = (2 * W0 - 1) * predict(density_k0, newdata=dataset, type="response") - W0 + 1,
      predicted_density_k1 = (2 * W1 - 1) * predict(density_k1, newdata=dataset, type="response") - W1 + 1,
      predicted_density_k2 = (2 * W2 - 1) * predict(density_k2, newdata=dataset, type="response") - W2 + 1,
      predicted_density = predicted_density_k0 * predicted_density_k1 * predicted_density_k2
    ) |>
    unique() |>
    mutate(
      gcomp = predicted_outcome * predicted_density
    ) |>
    summarise(
      gcomp = sum(gcomp)
    )
}

gcomp_multiple_time_points(never_treat)

gcomp_multiple_time_points(always_treat)


# Now, what was actually observed?
observed_density_k0 = unique_data$W0 * density_k0 + (1 - unique_data$W0) * (1 - density_k0)

observed_density_k1 = unique_data$W1 * density_k1 + (1 - unique_data$W1) * (1 - density_k1)

observed_density_k2 = unique_data$W2 * density_k2 + (1 - unique_data$W2) * (1 - density_k2)

# Now, multiply the densities together
observed_density = observed_density_k0 * observed_density_k1 * observed_density_k2


data |>
  mutate(
    # Get E[Y | ...]
    predicted_unique_data = predict(outcome_model, newdata=data, type="response"),
    # For each time point, estimate density under current
    density_k0 = predict(density_k0, newdata=data, type="response"),
    density_k1 = predict(density_k1, newdata=data, type="response"),
    density_k2 = predict(density_k2, newdata=data, type="response"),
    # Now,
    observed_density_k0 = W0 * density_k0 + (1-W0) * (1-density_k0),
    observed_density_k1 = W1 * density_k1 + (1-W1) * (1-density_k1),
    observed_density_k2 = W2 * density_k2 + (1-W2) * (1-density_k2),
    probability_of_observation = observed_density_k0 * observed_density_k1 * observed_density_k2,
    weighted_outcome = predicted_unique_data * probability_of_observation
  ) |>
  group_by(A0, A1, A2) |>
  summarise(
    gcomp = mean(weighted_outcome),
    vargcomp = var(weighted_outcome)
  )


