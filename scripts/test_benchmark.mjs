import { CompactGenderClassifier } from './classifier.mjs';

async function runBenchmark() {
  console.log('===============================================================');
  console.log('Node.js Sub-Millisecond Gender Classification Benchmark');
  console.log('===============================================================');

  const classifier = new CompactGenderClassifier();
  await classifier.init();

  const testCases = [
    // Clean First Names
    { text: 'Supratim', expected: 'MALE' },
    { text: 'Sayantika', expected: 'FEMALE' },
    { text: 'Alexander', expected: 'MALE' },
    { text: 'Priya', expected: 'FEMALE' },
    { text: 'John', expected: 'MALE' },
    { text: 'Emily', expected: 'FEMALE' },
    { text: 'Aarav', expected: 'MALE' },
    { text: 'Fatima', expected: 'FEMALE' },
    { text: 'Rahul', expected: 'MALE' },
    { text: 'Ananya', expected: 'FEMALE' },
    // Full Names (Given Name + Surnames)
    { text: 'Supratim Mondal', expected: 'MALE' },
    { text: 'Sayantika Biswas', expected: 'FEMALE' },
    { text: 'Alexander Hamilton', expected: 'MALE' },
    { text: 'Priya Sharma', expected: 'FEMALE' },
    { text: 'John Smith', expected: 'MALE' },
    { text: 'Emily Watson', expected: 'FEMALE' },
    { text: 'Dr. Rahul Verma', expected: 'MALE' },
    { text: 'Ms. Fatima Sheikh', expected: 'FEMALE' }
  ];

  // Warmup
  for (let i = 0; i < 20; i++) {
    await classifier.predict('Warmup Test');
  }

  console.log('\n--- Accuracy & Evaluation Results ---');
  let correct = 0;
  const latencies = [];

  for (const tc of testCases) {
    const t0 = performance.now();
    const res = await classifier.predict(tc.text);
    const dt = performance.now() - t0;
    latencies.push(dt);

    const match = res.gender === tc.expected;
    if (match) correct++;
    const icon = match ? '✅' : '❌';

    const p = res.raw_probs;
    const rawStr = `Male: ${(p.male * 100).toFixed(1)}% | Female: ${(p.female * 100).toFixed(1)}%`;
    const given = res.surnames.length > 0 ? `[Given: ${res.given_name}]` : '';
    console.log(
      `${icon} ${tc.text.padEnd(24)} => ${res.gender.padEnd(8)} (${(res.confidence * 100).toFixed(1)}%) | ${rawStr.padEnd(28)} ${given.padEnd(16)} [${dt.toFixed(3)} ms]`
    );
  }

  // Latency benchmark (1,000 iterations on single thread)
  console.log('\n--- Latency Benchmark (1,000 Inferences on Single Thread) ---');
  const benchNames = ['Supratim Mondal', 'Sayantika Biswas', 'Alexander Hamilton', 'Priya Sharma'];
  const benchStart = performance.now();
  const N = 1000;
  for (let i = 0; i < N; i++) {
    await classifier.predict(benchNames[i % benchNames.length]);
  }
  const totalBenchTime = performance.now() - benchStart;
  const avgLatencyMs = totalBenchTime / N;

  console.log(`Total Time for ${N.toLocaleString()} inferences: ${totalBenchTime.toFixed(2)} ms`);
  console.log(`Average Latency per Name:              ${avgLatencyMs.toFixed(3)} ms`);
  console.log(`Throughput:                            ${Math.round(1000 / avgLatencyMs).toLocaleString()} queries/sec`);
  console.log(`Accuracy on Test Suite:                ${correct}/${testCases.length} (${((correct / testCases.length) * 100).toFixed(1)}%)`);
  console.log('===============================================================\n');
}

runBenchmark().catch(console.error);
