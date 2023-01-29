test_that("KM function works", {

  iosw_data = data.frame(ID=1:12,
                         t = c(2,3,4,5,5,5,1,2,3,4,5,5),
                         delta = c(1,0,1,0,0,0,1,0,1,1,1,0),
                         W=rep(0:1, each=6))

  results = tibble::tibble(t = 1:5,
                       n = c(12,11,9,7,5),
                       events = c(1,1,1,2,1),
                       faux.hazard = c(0.9166667, 0.9090909, 0.8888889, 0.7142857, 0.8000000),
                       survival = c(0.9166667, 0.8333333, 0.7407407, 0.5291005, 0.4232804))

  expect_equal(kaplan_meier(iosw_data, ID="ID", t="t", delta="delta"),
               results,
               tolerance = 1e-5)
})
