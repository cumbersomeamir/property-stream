'use client';

import { useEffect, useState } from 'react';

interface Prediction {
  rank: number;
  ward_name: string;
  ward_no: string;
  zone: string;
  commercial_growth_rate: number;
  model1_prediction: number;
  model2_prediction?: number;
  households: number;
}

interface PredictionData {
  success: boolean;
  total_wards: number;
  model1_accuracy: { test_r2: number; cv_r2: number };
  model2_accuracy?: { test_r2: number; cv_r2: number };
  predictions: Prediction[];
}

export default function CommercialGrowthPage() {
  const [data, setData] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [selectedZone, setSelectedZone] = useState('all');

  useEffect(() => {
    async function loadPredictions() {
      try {
        const response = await fetch('/api/commercial-growth');
        const json = await response.json();
        setData(json);
      } catch (error) {
        console.error('Error loading predictions:', error);
      } finally {
        setLoading(false);
      }
    }
    loadPredictions();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-xl">Loading commercial growth predictions...</div>
      </div>
    );
  }

  if (!data || !data.success) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-xl text-red-600">Error loading predictions</div>
      </div>
    );
  }

  // Filter predictions
  const filtered = data.predictions.filter(p => {
    const matchesSearch = search === '' || 
      p.ward_name.toLowerCase().includes(search.toLowerCase()) ||
      p.ward_no.includes(search) ||
      p.zone.toLowerCase().includes(search.toLowerCase());
    const matchesZone = selectedZone === 'all' || p.zone.toLowerCase().includes(selectedZone.toLowerCase());
    return matchesSearch && matchesZone;
  });

  const uniqueZones = Array.from(new Set(data.predictions.map(p => p.zone))).sort();

  const getGrowthColor = (rate: number) => {
    if (rate >= 20) return 'bg-green-600 text-white';
    if (rate >= 10) return 'bg-green-500 text-white';
    if (rate >= 5) return 'bg-yellow-500 text-white';
    if (rate >= 0) return 'bg-orange-500 text-white';
    return 'bg-red-500 text-white';
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Commercial Growth Predictions
          </h1>
          <p className="text-lg text-gray-600 mb-4">
            ML-powered predictions for commercial growth rate by ward/zone
          </p>
          
          {/* Model Accuracy */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-white rounded-lg shadow-md p-4">
              <h3 className="font-semibold text-gray-700 mb-2">Model 1: Gradient Boosting</h3>
              <div className="text-sm text-gray-600">
                <div>Test R²: <span className="font-semibold">{(data.model1_accuracy.test_r2 * 100).toFixed(2)}%</span></div>
                <div>CV R²: <span className="font-semibold">{(data.model1_accuracy.cv_r2 * 100).toFixed(2)}%</span></div>
              </div>
            </div>
            {data.model2_accuracy && (
              <div className="bg-white rounded-lg shadow-md p-4">
                <h3 className="font-semibold text-gray-700 mb-2">Model 2: Deep Neural Network</h3>
                <div className="text-sm text-gray-600">
                  <div>Test R²: <span className="font-semibold">{(data.model2_accuracy.test_r2 * 100).toFixed(2)}%</span></div>
                  <div>CV R²: <span className="font-semibold">{(data.model2_accuracy.cv_r2 * 100).toFixed(2)}%</span></div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Filters */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Search</label>
              <input
                type="text"
                placeholder="Search by ward name, zone, or ward number..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Zone</label>
              <select
                value={selectedZone}
                onChange={(e) => setSelectedZone(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Zones</option>
                {uniqueZones.map(zone => (
                  <option key={zone} value={zone}>{zone}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="mt-4 text-sm text-gray-600">
            Showing {filtered.length} of {data.total_wards} wards
          </div>
        </div>

        {/* Predictions Table */}
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-800 text-white">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">Rank</th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">Ward Name</th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">Zone</th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">Growth Rate (%)</th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">Model 1</th>
                  {data.model2_accuracy && (
                    <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">Model 2</th>
                  )}
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">Households</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filtered.map((pred) => (
                  <tr key={`${pred.ward_name}-${pred.ward_no}`} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="font-bold text-gray-900">#{pred.rank}</span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="text-sm font-medium text-gray-900">{pred.ward_name}</div>
                      <div className="text-sm text-gray-500">Ward #{pred.ward_no}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                      {pred.zone}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-3 py-1 rounded-full text-sm font-semibold ${getGrowthColor(pred.commercial_growth_rate)}`}>
                        {pred.commercial_growth_rate > 0 ? '+' : ''}{pred.commercial_growth_rate.toFixed(2)}%
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                      {pred.model1_prediction > 0 ? '+' : ''}{pred.model1_prediction.toFixed(2)}%
                    </td>
                    {data.model2_accuracy && pred.model2_prediction !== undefined && (
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                        {pred.model2_prediction > 0 ? '+' : ''}{pred.model2_prediction.toFixed(2)}%
                      </td>
                    )}
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                      {pred.households.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}


