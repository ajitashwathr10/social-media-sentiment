async function analyzeSentiment() {
    const keyword = document.getElementById('keyword').value;
    if(!keyword) {
        return;
    }
    const response = await fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({keyword})
    });

    const data = await response.json();
    if(data.error) {
        alert(data.error);
        return;
    }

    document.getElementById('results').style.display = 'block';

    new Chart(document.getElementById('sentimentPie'), {
        type: 'pie',
        data: {
            labels: Object.keys(data.sentiment_counts),
            datasets: [{
                data: Object.values(data.sentiment_counts),
                backgroundColor: ['#4CAF50', '9E9E9E', '#F44336']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Sentiment Distribution'
                }
            }
        }
    });

    new Chart(document.getElementById('timeSeriesChart'), {
        type: 'line',
        data: {
            labels: data.time_series.labels,
            datasets: [
                {
                    label: 'Positive',
                    data: data.time_series.positive,
                    borderColor: '#4CAF50',
                    fill: false
                },
                {
                    label: 'Neutral',
                    data: data.time_series.neutral,
                    borderColor: '#9E9E9E',
                    fill: false
                },
                {
                    label: 'Negative',
                    data: data.time_series.negative,
                    borderColor: '#F44336',
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Sentiment Over Time'
                }
            }
        }
    });

    const tweetsContainer = document.getElementById('tweets');
    tweetsContainer.innerHTML = data.tweets.map(tweet => 
        `<div class = "tweet ${tweet.sentiment}">
            <strong>@${tweet.username}</strong>
            <p>${tweet.text}</p>
            <small>Sentiment: ${tweet.sentiment} | Followers: ${tweet.followers}</small>
        </div>`
    ).join('');
}