import pandas as pd


def calculate_qs_kendall_responsiveness(data):
    responsiveness_weights = {
        'open_to_total_pulls_ratio': 0.5,
        'open_to_total_issues_ratio': 0.5
    }
    responsiveness_signs = {
        'open_to_total_pulls_ratio': -1,
        'open_to_total_issues_ratio': -1
    }

    n = len(data['open_to_total_pulls_ratio'])
    responsiveness_scores = [0] * n

    for var in responsiveness_weights:
        for i in range(n):
            responsiveness_scores[i] += (responsiveness_signs[var] * responsiveness_weights[var] * data[var][i])

    return responsiveness_scores


def calculate_qs_kendall_popularity(data):
    popularity_weights = {
        'network_events': 0.2,
        'forks': 0.2,
        'subscribers': 0.2,
        'watchers': 0.2,
        'stargazers': 0.2
    }
    popularity_signs = {
        'network_events': 1,
        'forks': 1,
        'subscribers': 1,
        'watchers': 1,
        'stargazers': 1
    }

    n = len(data['network_events'])
    popularity_scores = [0] * n

    for var in popularity_weights:
        for i in range(n):
            popularity_scores[i] += (popularity_signs[var] * popularity_weights[var] * data[var][i])

    return popularity_scores


def calculate_qs_kendall_efficiency(data):
    efficiency_weights = {
        'creation_date': 0.2,
        'commits': 0.2,
        'latest_release': 0.2,
        'releases': 0.2,
        'latest_release': 0.2
    }

    efficiency_signs = {
        'creation_date': -1,
        'commits': 1,
        'latest_release': 1,
        'releases': 1,
        'latest_release': 1
    }

    n = len(data['creation_date'])
    efficiency_scores = [0] * n

    for var in efficiency_weights:
        for i in range(n):
            efficiency_scores[i] += (efficiency_signs[var] * efficiency_weights[var] * data[var][i])

    return efficiency_scores


def calculate_qs_spearman_popularity(data):
    popularity_weights = {
        'network_events': 0.2,
        'forks': 0.2,
        'subscribers': 0.2,
        'watchers': 0.2,
        'stargazers': 0.2
    }
    popularity_signs = {
        'network_events': 1,
        'forks': 1,
        'subscribers': 1,
        'watchers': 1,
        'stargazers': 1
    }

    n = len(data['network_events'])
    popularity_scores = [0] * n

    for var in popularity_weights:
        for i in range(n):
            popularity_scores[i] += (popularity_signs[var] * popularity_weights[var] * data[var][i])

    return popularity_scores


def calculate_qs_spearman_responsiveness(data):
    responsiveness_weights = {
        'open_to_total_pulls_ratio': 0.5,
        'open_to_total_issues_ratio': 0.5
    }
    responsiveness_signs = {
        'open_to_total_pulls_ratio': -1,
        'open_to_total_issues_ratio': -1
    }

    n = len(data['open_to_total_pulls_ratio'])
    responsiveness_scores = [0] * n

    for var in responsiveness_weights:
        for i in range(n):
            responsiveness_scores[i] += (responsiveness_signs[var] * responsiveness_weights[var] * data[var][i])

    return responsiveness_scores


def calculate_qs_spearman_efficiency(data):
    efficiency_weights = {
        'creation_date': 0.2,
        'commits': 0.2,
        'latest_release': 0.2,
        'releases': 0.2,
        'latest_release': 0.2
    }

    efficiency_signs = {
        'creation_date': -1,
        'commits': 1,
        'latest_release': 1,
        'releases': 1,
        'latest_release': 1
    }

    n = len(data['creation_date'])
    efficiency_scores = [0] * n

    for var in efficiency_weights:
        for i in range(n):
            efficiency_scores[i] += (efficiency_signs[var] * efficiency_weights[var] * data[var][i])

    return efficiency_scores


def calculate_qs_pearson_health(data):
    health_weights = {
        'open_to_total_pulls_ratio': 0.2,
        'open_to_total_issues_ratio': 0.2,
        'latest_release': 0.2,
        'creation_date': 0.2,
        'releases': 0.2
    }
    health_signs = {
        'open_to_total_pulls_ratio': -1,
        'open_to_total_issues_ratio': -1,
        'latest_release': 1,
        'creation_date': -1,
        'releases': 1
    }

    n = len(data['open_to_total_pulls_ratio'])
    health_scores = [0] * n

    for var in health_weights:
        for i in range(n):
            health_scores[i] += (health_signs[var] * health_weights[var] * data[var][i])

    return health_scores


def calculate_qs_pearson_engagement(data):
    weights = {
        'contributors': 0.14285714285714285,
        'commits': 0.14285714285714285,
        'network_events': 0.14285714285714285,
        'forks': 0.14285714285714285,
        'subscribers': 0.14285714285714285,
        'watchers': 0.14285714285714285,
        'stargazers': 0.14285714285714285
    }
    signs = {
        'contributors': 1,
        'commits': 1,
        'network_events': 1,
        'forks': 1,
        'subscribers': 1,
        'watchers': 1,
        'stargazers': 1
    }

    n = len(data['contributors'])
    scores = [0] * n

    for var in weights:
        for i in range(n):
            scores[i] += (signs[var] * weights[var] * data[var][i])

    return scores
