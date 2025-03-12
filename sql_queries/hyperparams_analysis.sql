SELECT 
    p.key AS hyperparam, 
    p.value AS param_value, 
    AVG(m.value) AS avg_rmse
FROM runs r
JOIN params p ON r.run_uuid = p.run_uuid
JOIN metrics m ON r.run_uuid = m.run_uuid
WHERE m.key = 'RMSE'
GROUP BY p.key, p.value
ORDER BY avg_rmse ASC;