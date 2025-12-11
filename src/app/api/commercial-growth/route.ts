import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export async function GET() {
  try {
    const scriptPath = path.join(process.cwd(), 'ml', 'predict_commercial_growth.py');
    
    const pythonProcess = spawn('python3', [scriptPath]);
    
    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

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

    // Parse JSON output (skip warnings)
    const jsonMatch = stdout.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      return NextResponse.json(
        { success: false, error: 'No JSON output found' },
        { status: 500 }
      );
    }

    const result = JSON.parse(jsonMatch[0]);
    return NextResponse.json(result);
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}

