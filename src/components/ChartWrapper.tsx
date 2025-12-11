'use client';

interface ChartWrapperProps {
  title: string;
  children: React.ReactNode;
  source: string;
  insight?: string;
}

export default function ChartWrapper({ title, children, source, insight }: ChartWrapperProps) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-gray-800">{title}</h3>
      {insight && (
        <div className="mb-4 p-3 bg-blue-50 border-l-4 border-blue-500 rounded">
          <p className="text-sm text-gray-700">{insight}</p>
        </div>
      )}
      <div className="relative min-h-[300px]">
        {children}
        <div className="absolute bottom-2 right-4 text-xs text-gray-500 bg-white/90 px-2 py-1 rounded shadow-sm z-10">
          Source: {source}
        </div>
      </div>
    </div>
  );
}

