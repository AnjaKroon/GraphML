In this folder there are two types of data.

Type 1: unnormalized data.
  data: X_unnormalized
      Dim1: dates, 30 days in total
      Dim2: nodes, all the locations
      Dim3: features (population, weather, matter)
      
  ground truth: confirmed_unnormalized
      Dim1: dates, 30 days in total
      Dim2: nodes, all the locations


Type 2: normalized data.
  data: X_normalized
      Dim1: dates, 30 days in total
      Dim2: nodes, all the locations
      Dim3: features (population, weather, matter)
      
  ground truth: confirmed_normalized
      Dim1: dates, 30 days in total
      Dim2: nodes, all the locations


population: PCA of 'population',
                   'population_age_80_and_older', 
                   'population_age_70_79',
                   'population_age_60_69',
                   'population_age_50_59',
                   'population_age_30_39',
                   'population_age_40_49',
                   'population_age_00_09',
                   'population_female',
                   'population_male',
                   'population_age_20_29',
                   'population_age_10_19',
                   'life_expectancy'
                   (or PCA of normalized version of the above features)

weather: PCA of 'average_temperature_celsius', 
                'minimum_temperature_celsius',
                'rainfall_mm',
                'maximum_temperature_celsius',
                'relative_humidity',
                'dew_point'
                (or PCA of normalized version of the above features)

matter: consists of 'cumulative_confirmed',
                    'cumulative_deceased',
                    'new_deceased',
                    'cumulative_persons_fully_vaccinated',
                    'new_persons_fully_vaccinated'
                    (or normalized version of the above features)

confirmed: 'new_confirmed' (or its normalized version)

locations_data_unique: locations_data_unique[i] corresponds to population[:, i, :](the ith node).
                    

                   

      
