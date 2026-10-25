import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const SCREENSHOT_DIR = path.resolve('test-results/screenshots');

// Ensure screenshots directory exists
test.beforeAll(async () => {
  if (!fs.existsSync(SCREENSHOT_DIR)) {
    fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
  }
});

test.describe('DiaPredict End-to-End Application Tests', () => {

  test('Test 1: Application Loads', async ({ page }) => {
    // 1. Open DiaPredict frontend
    await page.goto('/');

    // 2. Verify page title
    await expect(page).toHaveTitle(/DiaPredict/);

    // 3. Verify header title and subtitle
    const appHeading = page.getByRole('heading', { name: 'DiaPredict', exact: true });
    await expect(appHeading).toBeVisible();
    await expect(page.getByText('AI-Powered Diabetes Prediction')).toBeVisible();

    // 4. Verify main diabetes assessment form and sections
    await expect(page.getByRole('heading', { name: 'Patient Diagnostic Parameters' })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Prediction Result' })).toBeVisible();

    // 5. Verify model metrics section loads actual data from backend
    await expect(page.getByText('Model Performance & Evaluation Metrics')).toBeVisible();
    // Accuracy should load and display with percentage
    const accuracyLabel = page.getByText('Accuracy', { exact: true });
    await expect(accuracyLabel).toBeVisible();
    const metricsGrid = page.locator('.metrics-grid');
    await expect(metricsGrid).toBeVisible();
    await expect(page.getByText('Total Dataset:')).toBeVisible();

    // 6. Capture full-page screenshot
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '01-homepage.png'),
      fullPage: true,
    });
  });

  test('Test 2: Successful Diabetes Prediction', async ({ page }) => {
    // 1. Open the application
    await page.goto('/');

    // 2. Fill in all 8 numeric fields using getByLabel
    await page.getByLabel(/Pregnancies/).fill('2');
    await page.getByLabel(/Glucose Level/).fill('140');
    await page.getByLabel(/Blood Pressure/).fill('80');
    await page.getByLabel(/Skin Thickness/).fill('25');
    await page.getByLabel(/2-Hour Serum Insulin/).fill('100');
    await page.getByLabel(/Body Mass Index/).fill('28.5');
    await page.getByLabel(/Diabetes Pedigree Function/).fill('0.63');
    await page.getByLabel(/Age/).fill('35');

    // 3. Set up listener to confirm real backend prediction endpoint is invoked
    const predictPromise = page.waitForResponse(
      (resp) => resp.url().includes('/api/predict') && resp.status() === 200
    );

    // 4. Click Predict button
    const predictBtn = page.getByRole('button', { name: 'Predict Diabetes Risk' });
    await predictBtn.click();

    // 5. Wait for the actual Flask backend response
    const predictResponse = await predictPromise;
    const responseJson = await predictResponse.json();

    // Verify backend returned valid classification and probability
    expect([0, 1]).toContain(responseJson.prediction);
    expect(responseJson.probability).toBeGreaterThanOrEqual(0);
    expect(responseJson.probability).toBeLessThanOrEqual(1);

    // 6. Verify UI prediction panel displays result
    const outcomeBox = page.locator('.prediction-outcome');
    await expect(outcomeBox).toBeVisible();

    // Check classification label matches backend output
    await expect(outcomeBox).toContainText(`Classification Class: ${responseJson.prediction}`);

    // Verify probability percentage and progress bar
    const probHeader = page.locator('.probability-header');
    await expect(probHeader).toBeVisible();
    const probText = await probHeader.textContent();

    // Extract probability percentage number and verify it's a finite number between 0 and 100
    const match = probText.match(/(\d+(\.\d+)?)%/);
    expect(match).not.toBeNull();
    const percentValue = parseFloat(match[1]);
    expect(percentValue).toBeGreaterThanOrEqual(0);
    expect(percentValue).toBeLessThanOrEqual(100);

    const progressBar = page.locator('.progress-fill');
    await expect(progressBar).toBeVisible();

    // 7. Capture screenshot of the completed prediction
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '02-successful-prediction.png'),
      fullPage: true,
    });
  });

  test('Test 3: Form Validation', async ({ page }) => {
    // 1. Open application
    await page.goto('/');

    // 2. Leave fields empty and attempt to submit
    const predictBtn = page.getByRole('button', { name: 'Predict Diabetes Risk' });
    await predictBtn.click();

    // 3. Verify no prediction result is generated and empty state persists
    await expect(page.getByText('No prediction yet')).toBeVisible();
    await expect(page.locator('.prediction-outcome')).not.toBeVisible();

    // 4. Check that at least one required input is flagged invalid by HTML5 validity
    const firstInput = page.getByLabel(/Pregnancies/);
    const isValid = await firstInput.evaluate((el) => el.checkValidity());
    expect(isValid).toBe(false);

    // 5. Capture screenshot of validation state
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '03-form-validation.png'),
      fullPage: true,
    });
  });

  test('Test 4: Reset Functionality', async ({ page }) => {
    // 1. Fill all 8 fields and submit
    await page.goto('/');

    await page.getByLabel(/Pregnancies/).fill('3');
    await page.getByLabel(/Glucose Level/).fill('125');
    await page.getByLabel(/Blood Pressure/).fill('74');
    await page.getByLabel(/Skin Thickness/).fill('22');
    await page.getByLabel(/2-Hour Serum Insulin/).fill('90');
    await page.getByLabel(/Body Mass Index/).fill('26.4');
    await page.getByLabel(/Diabetes Pedigree Function/).fill('0.45');
    await page.getByLabel(/Age/).fill('29');

    const predictPromise = page.waitForResponse(
      (resp) => resp.url().includes('/api/predict') && resp.status() === 200
    );
    await page.getByRole('button', { name: 'Predict Diabetes Risk' }).click();
    await predictPromise;

    // Verify prediction outcome appeared
    await expect(page.locator('.prediction-outcome')).toBeVisible();

    // 2. Click Reset button
    const resetBtn = page.getByRole('button', { name: 'Reset' });
    await resetBtn.click();

    // 3. Verify all inputs are cleared
    await expect(page.getByLabel(/Pregnancies/)).toHaveValue('');
    await expect(page.getByLabel(/Glucose Level/)).toHaveValue('');
    await expect(page.getByLabel(/Blood Pressure/)).toHaveValue('');
    await expect(page.getByLabel(/Skin Thickness/)).toHaveValue('');
    await expect(page.getByLabel(/2-Hour Serum Insulin/)).toHaveValue('');
    await expect(page.getByLabel(/Body Mass Index/)).toHaveValue('');
    await expect(page.getByLabel(/Diabetes Pedigree Function/)).toHaveValue('');
    await expect(page.getByLabel(/Age/)).toHaveValue('');

    // 4. Verify prediction result returns to initial empty state
    await expect(page.getByText('No prediction yet')).toBeVisible();
    await expect(page.locator('.prediction-outcome')).not.toBeVisible();

    // 5. Capture screenshot of reset form
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '04-reset-form.png'),
      fullPage: true,
    });
  });

  test('Test 5: Multiple Predictions', async ({ page }) => {
    await page.goto('/');

    // 1. First set of inputs (Lower risk profile)
    await page.getByLabel(/Pregnancies/).fill('1');
    await page.getByLabel(/Glucose Level/).fill('85');
    await page.getByLabel(/Blood Pressure/).fill('66');
    await page.getByLabel(/Skin Thickness/).fill('20');
    await page.getByLabel(/2-Hour Serum Insulin/).fill('60');
    await page.getByLabel(/Body Mass Index/).fill('22.0');
    await page.getByLabel(/Diabetes Pedigree Function/).fill('0.25');
    await page.getByLabel(/Age/).fill('24');

    const firstRespPromise = page.waitForResponse(
      (resp) => resp.url().includes('/api/predict') && resp.status() === 200
    );
    await page.getByRole('button', { name: 'Predict Diabetes Risk' }).click();
    const firstResp = await firstRespPromise;
    const firstData = await firstResp.json();

    const firstProbText = await page.locator('.probability-header strong').textContent();

    // 2. Change inputs to higher risk profile
    await page.getByLabel(/Pregnancies/).fill('7');
    await page.getByLabel(/Glucose Level/).fill('180');
    await page.getByLabel(/Blood Pressure/).fill('88');
    await page.getByLabel(/Skin Thickness/).fill('38');
    await page.getByLabel(/2-Hour Serum Insulin/).fill('190');
    await page.getByLabel(/Body Mass Index/).fill('36.5');
    await page.getByLabel(/Diabetes Pedigree Function/).fill('0.95');
    await page.getByLabel(/Age/).fill('55');

    const secondRespPromise = page.waitForResponse(
      (resp) => resp.url().includes('/api/predict') && resp.status() === 200
    );
    await page.getByRole('button', { name: 'Predict Diabetes Risk' }).click();
    const secondResp = await secondRespPromise;
    const secondData = await secondResp.json();

    // 3. Verify second response is distinct and UI updated accordingly
    expect(secondData.probability).not.toEqual(firstData.probability);

    const secondProbText = await page.locator('.probability-header strong').textContent();
    expect(secondProbText).not.toEqual(firstProbText);

    // 4. Capture screenshot of updated prediction
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '05-updated-prediction.png'),
      fullPage: true,
    });
  });

});
