SELECT 
    e.experiment_id, 
    e.name AS experiment_name, 
    COUNT(r.run_uuid) AS total_runs
FROM experiments e
LEFT JOIN runs r ON e.experiment_id = r.experiment_id
GROUP BY e.experiment_id, e.name
ORDER BY total_runs DESC;