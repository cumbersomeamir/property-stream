'use client';

import { useState, useEffect } from 'react';

interface WardPrediction {
  ward_name: string;
  ward_no: string;
  zone: string;
  zone_num: number;
  households: number;
  investment_score: number;
  rental_yield: number;
  annual_appreciation_rate: number;
  roi_3yr: number;
  risk_score: number;
  recommendation: string;
  rank: number;
}

export default function PredictPage() {
  const [predictions, setPredictions] = useState<WardPrediction[]>([]);
  const [filteredPredictions, setFilteredPredictions] = useState<WardPrediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortConfig, setSortConfig] = useState<{ key: keyof WardPrediction; direction: 'asc' | 'desc' }>({
    key: 'rank',
    direction: 'asc'
  });
  const [filters, setFilters] = useState({
    recommendation: 'all',
    minScore: '',
    maxRisk: '',
    zone: 'all',
    search: '',
    selectedWard: 'all'
  });
  const [selectedWardDetail, setSelectedWardDetail] = useState<WardPrediction | null>(null);

  useEffect(() => {
    async function loadPredictions() {
      try {
        const response = await fetch('/all_predictions.json');
        const data = await response.json();
        if (data.success) {
          setPredictions(data.predictions);
          setFilteredPredictions(data.predictions);
        }
      } catch (err) {
        console.error('Error loading predictions:', err);
        // Try loading from API instead
        try {
          const response = await fetch('/api/all_predictions');
          const data = await response.json();
          if (data.success) {
            setPredictions(data.predictions);
            setFilteredPredictions(data.predictions);
          }
        } catch (apiErr) {
          console.error('API also failed:', apiErr);
        }
      } finally {
        setLoading(false);
      }
    }
    loadPredictions();
  }, []);

  useEffect(() => {
    let filtered = [...predictions];

    // Filter by recommendation
    if (filters.recommendation !== 'all') {
      filtered = filtered.filter(p => p.recommendation === filters.recommendation);
    }

    // Filter by minimum score
    if (filters.minScore) {
      const min = parseFloat(filters.minScore);
      filtered = filtered.filter(p => p.investment_score >= min);
    }

    // Filter by maximum risk
    if (filters.maxRisk) {
      const max = parseFloat(filters.maxRisk);
      filtered = filtered.filter(p => p.risk_score <= max);
    }

    // Filter by zone
    if (filters.zone !== 'all') {
      filtered = filtered.filter(p => p.zone_num.toString() === filters.zone);
    }

    // Filter by selected ward dropdown
    if (filters.selectedWard !== 'all') {
      filtered = filtered.filter(p => p.ward_name === filters.selectedWard);
    }

    // Filter by search (works independently)
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      filtered = filtered.filter(p => 
        p.ward_name.toLowerCase().includes(searchLower) ||
        p.zone.toLowerCase().includes(searchLower) ||
        p.ward_no.toLowerCase().includes(searchLower)
      );
    }

    // Apply sorting
    filtered.sort((a, b) => {
      const aVal = a[sortConfig.key];
      const bVal = b[sortConfig.key];
      
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortConfig.direction === 'asc' ? aVal - bVal : bVal - aVal;
      }
      return sortConfig.direction === 'asc' 
        ? String(aVal).localeCompare(String(bVal))
        : String(bVal).localeCompare(String(aVal));
    });

    setFilteredPredictions(filtered);
  }, [predictions, filters, sortConfig]);

  const handleSort = (key: keyof WardPrediction) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'desc' ? 'asc' : 'desc'
    }));
  };

  const getRecommendationColor = (rec: string) => {
    switch (rec) {
      case 'STRONG BUY': return 'bg-green-600 text-white';
      case 'BUY': return 'bg-green-500 text-white';
      case 'HOLD': return 'bg-yellow-500 text-white';
      case 'AVOID': return 'bg-red-500 text-white';
      default: return 'bg-gray-500 text-white';
    }
  };

  const getScoreColor = (score: number, max: number = 100) => {
    const percentage = (score / max) * 100;
    if (percentage >= 70) return 'text-green-600 font-bold';
    if (percentage >= 50) return 'text-yellow-600 font-semibold';
    return 'text-red-600';
  };

  const uniqueZones = Array.from(new Set(predictions.map(p => p.zone_num))).sort();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-xl">Loading predictions...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Property Investment Predictions - All Wards
          </h1>
          <p className="text-lg text-gray-600">
            ML-powered investment scores for all {predictions.length} wards in Lucknow
          </p>
        </div>

        {/* Filters */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Filters & Search</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
            <div className="lg:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">Search</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="Search by ward name, zone, or ward number..."
                  value={filters.search}
                  onChange={(e) => setFilters({...filters, search: e.target.value})}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      // Search happens automatically on filter change
                    }
                  }}
                  className="flex-1 p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <button
                  onClick={() => {
                    // Trigger re-filter by updating state
                    setFilters({...filters});
                  }}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold transition-colors"
                >
                  Search
                </button>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Select Ward</label>
              <select
                value={filters.selectedWard}
                onChange={(e) => setFilters({...filters, selectedWard: e.target.value})}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="all">All Wards</option>
                {predictions
                  .sort((a, b) => a.ward_name.localeCompare(b.ward_name))
                  .map(ward => (
                    <option key={ward.ward_name} value={ward.ward_name}>
                      {ward.ward_name} (Zone {ward.zone_num})
                    </option>
                  ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Recommendation</label>
              <select
                value={filters.recommendation}
                onChange={(e) => setFilters({...filters, recommendation: e.target.value})}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="all">All</option>
                <option value="STRONG BUY">STRONG BUY</option>
                <option value="BUY">BUY</option>
                <option value="HOLD">HOLD</option>
                <option value="AVOID">AVOID</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Min Investment Score</label>
              <input
                type="number"
                placeholder="e.g. 30"
                value={filters.minScore}
                onChange={(e) => setFilters({...filters, minScore: e.target.value})}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Max Risk Score</label>
              <input
                type="number"
                placeholder="e.g. 5"
                value={filters.maxRisk}
                onChange={(e) => setFilters({...filters, maxRisk: e.target.value})}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4 mt-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Zone</label>
              <select
                value={filters.zone}
                onChange={(e) => setFilters({...filters, zone: e.target.value})}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="all">All Zones</option>
                {uniqueZones.map(zone => (
                  <option key={zone} value={zone.toString()}>Zone {zone}</option>
                ))}
              </select>
            </div>
            <div className="lg:col-span-5 flex items-end">
              <button
                onClick={() => {
                  setFilters({
                    recommendation: 'all',
                    minScore: '',
                    maxRisk: '',
                    zone: 'all',
                    search: '',
                    selectedWard: 'all'
                  });
                }}
                className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 font-semibold transition-colors"
              >
                Clear All Filters
              </button>
            </div>
          </div>
          <div className="mt-4 text-sm text-gray-600">
            Showing {filteredPredictions.length} of {predictions.length} wards
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-lg shadow-md p-4">
            <div className="text-sm text-gray-600">Top Score</div>
            <div className="text-2xl font-bold text-green-600">
              {predictions[0]?.investment_score || 0}/100
            </div>
            <div className="text-xs text-gray-500">{predictions[0]?.ward_name}</div>
          </div>
          <div className="bg-white rounded-lg shadow-md p-4">
            <div className="text-sm text-gray-600">STRONG BUY</div>
            <div className="text-2xl font-bold text-green-600">
              {predictions.filter(p => p.recommendation === 'STRONG BUY').length}
            </div>
            <div className="text-xs text-gray-500">wards</div>
          </div>
          <div className="bg-white rounded-lg shadow-md p-4">
            <div className="text-sm text-gray-600">Avg Investment Score</div>
            <div className="text-2xl font-bold">
              {(predictions.reduce((sum, p) => sum + p.investment_score, 0) / predictions.length).toFixed(1)}
            </div>
            <div className="text-xs text-gray-500">out of 100</div>
          </div>
          <div className="bg-white rounded-lg shadow-md p-4">
            <div className="text-sm text-gray-600">Avg 3-Year ROI</div>
            <div className="text-2xl font-bold text-blue-600">
              {(predictions.reduce((sum, p) => sum + p.roi_3yr, 0) / predictions.length).toFixed(1)}%
            </div>
            <div className="text-xs text-gray-500">expected return</div>
          </div>
        </div>

        {/* Table */}
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('rank')}>
                    Rank {sortConfig.key === 'rank' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('ward_name')}>
                    Ward Name {sortConfig.key === 'ward_name' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('zone')}>
                    Zone {sortConfig.key === 'zone' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('investment_score')}>
                    Investment Score {sortConfig.key === 'investment_score' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('rental_yield')}>
                    Rental Yield {sortConfig.key === 'rental_yield' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('annual_appreciation_rate')}>
                    Appreciation {sortConfig.key === 'annual_appreciation_rate' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('roi_3yr')}>
                    3-Year ROI {sortConfig.key === 'roi_3yr' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('risk_score')}>
                    Risk {sortConfig.key === 'risk_score' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('recommendation')}>
                    Recommendation {sortConfig.key === 'recommendation' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Details
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredPredictions.map((ward) => (
                  <tr key={ward.ward_name} className="hover:bg-gray-50">
                    <td className="px-4 py-3 whitespace-nowrap">
                      <span className="font-bold text-gray-900">#{ward.rank}</span>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{ward.ward_name}</div>
                      <div className="text-xs text-gray-500">Ward #{ward.ward_no} • {ward.households.toLocaleString()} HH</div>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600">
                      {ward.zone}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <span className={`text-lg font-bold ${getScoreColor(ward.investment_score)}`}>
                        {ward.investment_score}
                      </span>
                      <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                        <div
                          className={`h-2 rounded-full ${
                            ward.investment_score >= 45 ? 'bg-green-600' :
                            ward.investment_score >= 35 ? 'bg-green-500' :
                            ward.investment_score >= 25 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${ward.investment_score}%` }}
                        />
                      </div>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm font-semibold text-blue-600">
                      {ward.rental_yield}%
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm font-semibold text-purple-600">
                      {ward.annual_appreciation_rate}%
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm font-bold text-green-600">
                      {ward.roi_3yr}%
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <span className={`text-sm font-semibold ${
                        ward.risk_score <= 3 ? 'text-green-600' :
                        ward.risk_score <= 5 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {ward.risk_score}/10
                      </span>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <span className={`px-2 py-1 text-xs font-semibold rounded-full ${getRecommendationColor(ward.recommendation)}`}>
                        {ward.recommendation}
                      </span>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <button
                        onClick={() => setSelectedWardDetail(ward)}
                        className="text-blue-600 hover:text-blue-800 font-semibold text-sm"
                      >
                        View Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Detailed View Modal */}
        {selectedWardDetail && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50" onClick={() => setSelectedWardDetail(null)}>
            <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className={`p-6 ${getRecommendationColor(selectedWardDetail.recommendation)}`}>
                <div className="flex justify-between items-start">
                  <div>
                    <h2 className="text-3xl font-bold mb-2 text-white drop-shadow-md">{selectedWardDetail.ward_name}</h2>
                    <p className="text-lg text-white drop-shadow-md opacity-95">{selectedWardDetail.zone} • Ward #{selectedWardDetail.ward_no}</p>
                  </div>
                  <button
                    onClick={() => setSelectedWardDetail(null)}
                    className="text-white text-3xl font-bold hover:opacity-75 drop-shadow-md w-8 h-8 flex items-center justify-center rounded-full hover:bg-white hover:bg-opacity-20"
                  >
                    ×
                  </button>
                </div>
              </div>
              
              <div className="p-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-6">
                  <div>
                    <div className="text-sm text-gray-600 mb-1">Investment Score</div>
                    <div className={`text-3xl font-bold ${getScoreColor(selectedWardDetail.investment_score)}`}>
                      {selectedWardDetail.investment_score}
                    </div>
                    <div className="text-xs text-gray-500">Rank #{selectedWardDetail.rank}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600 mb-1">Rental Yield</div>
                    <div className="text-3xl font-bold text-blue-600">{selectedWardDetail.rental_yield}%</div>
                    <div className="text-xs text-gray-500">Annual return</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600 mb-1">Appreciation</div>
                    <div className="text-3xl font-bold text-purple-600">{selectedWardDetail.annual_appreciation_rate}%</div>
                    <div className="text-xs text-gray-500">Per year</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600 mb-1">3-Year ROI</div>
                    <div className="text-3xl font-bold text-green-600">{selectedWardDetail.roi_3yr}%</div>
                    <div className="text-xs text-gray-500">Total return</div>
                  </div>
                </div>

                <div className="border-t pt-6">
                  <h3 className="text-xl font-semibold mb-4 text-gray-900">Investment Breakdown</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between py-2 border-b">
                      <span className="text-gray-700 font-medium">Initial Investment (₹1Cr)</span>
                      <span className="font-semibold text-gray-900">₹1,00,00,000</span>
                    </div>
                    <div className="flex justify-between py-2 border-b">
                      <span className="text-gray-700 font-medium">Annual Rental Income</span>
                      <span className="font-semibold text-green-600">
                        ₹{(selectedWardDetail.rental_yield * 100000).toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between py-2 border-b">
                      <span className="text-gray-700 font-medium">Annual Appreciation</span>
                      <span className="font-semibold text-blue-600">
                        ₹{(selectedWardDetail.annual_appreciation_rate * 100000).toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between py-2 border-b">
                      <span className="text-gray-700 font-medium">Total Annual Return</span>
                      <span className="font-semibold text-gray-900">
                        ₹{((selectedWardDetail.rental_yield + selectedWardDetail.annual_appreciation_rate) * 100000).toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between py-2 pt-4 border-t-2 border-gray-300">
                      <span className="font-bold text-lg text-gray-900">Property Value after 3 Years</span>
                      <span className="font-bold text-lg text-green-600">
                        ₹{((1 + selectedWardDetail.roi_3yr / 100) * 10000000).toLocaleString('en-IN')}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="mt-6 border-t pt-6">
                  <h3 className="text-xl font-semibold mb-4 text-gray-900">Risk Assessment</h3>
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-gray-700 font-medium">Risk Score</span>
                      <span className={`font-semibold text-lg ${
                        selectedWardDetail.risk_score <= 3 ? 'text-green-600' :
                        selectedWardDetail.risk_score <= 5 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {selectedWardDetail.risk_score}/10
                      </span>
                    </div>
                    <div className="h-4 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${
                          selectedWardDetail.risk_score <= 3 ? 'bg-green-500' :
                          selectedWardDetail.risk_score <= 5 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${selectedWardDetail.risk_score * 10}%` }}
                      />
                    </div>
                    <div className="mt-2 text-sm text-gray-500">
                      {selectedWardDetail.risk_score <= 3 ? 'Low Risk Investment' :
                       selectedWardDetail.risk_score <= 5 ? 'Medium Risk Investment' : 'High Risk Investment'}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
