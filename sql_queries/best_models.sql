SELECT 
    r.run_uuid, 
    e.name AS experiment_name, 
    m.value AS rmse
FROM runs r
JOIN experiments e ON r.experiment_id = e.experiment_id
JOIN metrics m ON r.run_uuid = m.run_uuid
WHERE m.key = 'RMSE'
ORDER BY m.value ASC 
LIMIT 3; 