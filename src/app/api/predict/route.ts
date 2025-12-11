import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const features = body.features;

    if (!features) {
      return NextResponse.json(
        { success: false, error: 'Features are required' },
        { status: 400 }
      );
    }

    // Get the path to the Python prediction script (using market-based model)
    const scriptPath = path.join(process.cwd(), 'ml', 'predict_market.py');
    
    // Spawn Python process
    const pythonProcess = spawn('python3', [scriptPath]);
    
    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    // Send input to Python process
    pythonProcess.stdin.write(JSON.stringify({ features }));
    pythonProcess.stdin.end();

    // Wait for process to complete
    const exitCode = await new Promise<number>((resolve) => {
      pythonProcess.on('close', (code) => {
        resolve(code || 0);
      });
    });

    if (exitCode !== 0) {
      return NextResponse.json(
        { success: false, error: `Python process failed: ${stderr}` },
        { status: 500 }
      );
    }

    // Parse Python output
    const result = JSON.parse(stdout);
    
    if (!result.success) {
      return NextResponse.json(
        { success: false, error: result.error },
        { status: 500 }
      );
    }

    return NextResponse.json(result);
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}

