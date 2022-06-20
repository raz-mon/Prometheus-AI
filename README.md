# Prometheus-AI
Applying classical and Deep-Learning algorithms to time-series data, collected through Prometheus (Thanos).

## Project goals
In this project, we wish to write an API that performs the following tasks:
 * Poll data from Prometheuse constantly. On this data we will use our classical and Deep-Learning algorithms.
 * Organize the data in an orderly fasion on the local computer running the queries (see file-paths before running).
 * Offer the user to pick the following parameters when performing forecasting:
   * Metric (cpu or memory in this case).
   * Application (from the applications monitored - see list in data_get.utils).
   * Forecasting method (See implemented methods in forecasting.Methods).
   * Test segment length (proportional to whole data-length).
   * Granularity of the data.
   * Compression method used to aggregate data-samples to the wanted granularity.
   * Error metric.

Other than these options, the user can also choose if he wants to:
  * Traverse the whole data-set, aggregating the error of a specific metric and application using a specific forecasting method.
  * Traverse the data-set, until finding one of the results received for specific metric and application, and plotting the forecast (and train period) of that data-sample (this option will also print the error).

## Introduction
Some introductory data, regarding the framework and technologies, and maybe some of the algorithms we will use (name-dropping at this stage.. Later will explain the methods in detail (in the docs or here at the main page)).

## Table of contents \ Useful links (one of the two):
Can put some link here to some usefull links, or arrange all data in docs, and send to the relevant one (See other README..).

 
