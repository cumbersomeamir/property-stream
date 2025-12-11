export default function Home() {
  return (
    <div className="min-h-screen p-8 bg-gray-50">
      <main className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold mb-4 text-gray-900">
          Property Stream
        </h1>
        <p className="text-lg text-gray-600 mb-8">
          Investment Recommendation Engine - Data Exploration
        </p>
        <div className="mt-8 space-x-4">
          <a 
            href="/lucknow" 
            className="inline-block bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors font-semibold"
          >
            View Lucknow Data Analysis →
          </a>
          <a 
            href="/predict" 
            className="inline-block bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 transition-colors font-semibold"
          >
            Property Investment Predictor →
          </a>
          <a 
            href="/commercial-growth" 
            className="inline-block bg-purple-600 text-white px-6 py-3 rounded-lg hover:bg-purple-700 transition-colors font-semibold"
          >
            Commercial Growth Predictions →
          </a>
        </div>
      </main>
    </div>
  );
}

